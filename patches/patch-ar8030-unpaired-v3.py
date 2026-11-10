#!/usr/bin/env python3
"""Patch the cx472 AR8030 baseband image for unpaired TX, v3.

v1/v2 (send body forced, scheduler descriptor enables kept) changed
nothing that mattered: the TX transport for user 0 is created at boot
and its send body runs, but the pull callback (c.jalr a2 at VA
0x235cac in bb_mac_tx_task) refuses to pull while the transport is
not slot-connected, and with no paired peer nothing ever connects it.
The host socket for port 2 stalls at tx_bytes=0.

The authentic connector is bb_mac_transport_set_slot_connected, which
exists only as a case inside bb_mac_tx_task's message dispatcher
(a0=1, a1=0x304, a2=msg, jump table at VA 0x254484 index 4 -> handler
VA 0x23674a; msg word0 = slot, msg byte5 == 2 means CONNECT).
bb_link_task normally posts it when a peer is heard. v3 keeps all
v1+v2 sites and synthesises that one message itself:

  bb_mac_tx_task's main case (a0=0/a1=0) ends by filling the per-user
  descriptor table; the fill loop's exit branch is at VA 0x2358d6
  (file 0x35ed6, orig bne a5, a1, +0x22 -> fill body VA 0x2358f8,
  fall through to the epilogue VA 0x2358da). v3 replaces that branch
  with a jump to a trampoline in a 38-byte alignment cave at VA
  0x24cf5a (file 0x4d55a, right after a c.jr ra):

    beq a5, a1, +8              ; fill finished?
    j   0x2358f8                ; no: keep filling (original branch)
    lui a2, 0x253               ; fake msg VA 0x2528b8
    addi a2, a2, -0x748         ;   0x253000 - 0x748 = 0x2528b8
    c.mv s2, a2                 ; handler expects the msg in s2 too
    j   0x23674a                ; bb_mac_transport_set_slot_connected

  The handler exits through the dispatcher's own epilogue, so no
  return path is needed; the descriptor fill is already complete at
  this point, so nothing is skipped. The fake msg is 8 zero bytes at
  VA 0x2528b8 (file 0x52eb8): word0 = slot 0 already, only byte5
  (file 0x52ebd) changes to 2 (CONNECT).

  The handler runs again on every main-case tick. That is deliberate
  (a guarded variant that walks the real pointer chain --
  *(*(0x26a61c)+4) byte 0 is slot 0's connected byte -- no longer
  fits the 38-byte cave) and harmless: it rewrites the same connected
  byte, reloads the same MAC/frame state and posts the same transport
  callbacks. In fact the first on-hardware v3 had a guard watching
  the wrong table entry, so this re-entry has been the running
  configuration for every successful test so far.

Hardware result (bench, v3 + json tweaks below): port 2 tx_bytes
climbs for the first time and the BR (user 8) MAC counters grow in
step -- in 1V1 the AP's downlink user data rides inside the BR
beacon frames' TLV payload, so there is no separate TX slot and the
host/link status keeps reporting IDLE/Dwdata 255 even while data
flows. With BR retx_count 100 the MAC's retx hold
(bb_link_retx_ctrl_feed) throttles the drain to ~1.2 KB/s; with
retx_count 0 it drains ~31 KB/s. RF reception of the BR frames is
still to be confirmed with a receiver/SDR.

Pair with (config, not blob): ar8030.json BR (line ~208) and slot
(line ~230) tx.retx_count -> 0, so a transmitted frame is not held
for an ACK that will never come. Keep 9 dBm and a fan.

This only accepts the cx472 blob shipped with builder 9f3ef6f
(md5 56225d53e7e8739b2c632beee2aaa9f0). Checksums are rewritten the
same way as v1/v2. Do not wire into the build until RF reception is
confirmed; recovery is restoring ar8030.img.orig.
"""

import hashlib
import struct
import sys
from pathlib import Path

EXPECTED_MD5 = "56225d53e7e8739b2c632beee2aaa9f0"
HEADER_LEN = 0x40
SPL_FILE_OFF = 0x4600
SPL_LEN_OFF = 0x1C
SPL_SUM_OFF = 0x34
NOP4 = bytes.fromhex("13000000")
BASE_VA = 0x204000


def va2off(va: int) -> int:
    return va - BASE_VA + SPL_FILE_OFF


# --- tiny RISC-V encoders (verified against capstone after building) ---

def enc_jal(rd: int, pc: int, target: int) -> bytes:
    imm = target - pc
    assert imm % 2 == 0 and -(1 << 20) <= imm < (1 << 20), hex(imm)
    u = imm & 0x1FFFFF
    word = (
        (((u >> 20) & 1) << 31)
        | (((u >> 1) & 0x3FF) << 21)
        | (((u >> 11) & 1) << 20)
        | (((u >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )
    return struct.pack("<I", word)


def enc_beq(rs1: int, rs2: int, pc: int, target: int) -> bytes:
    imm = target - pc
    assert imm % 2 == 0 and -(1 << 12) <= imm < (1 << 12), hex(imm)
    u = imm & 0x1FFF
    word = (
        (((u >> 12) & 1) << 31)
        | (((u >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0b000 << 12)
        | (((u >> 1) & 0xF) << 8)
        | (((u >> 11) & 1) << 7)
        | 0x63
    )
    return struct.pack("<I", word)


def enc_lui(rd: int, imm20: int) -> bytes:
    return struct.pack("<I", ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37)


def enc_addi(rd: int, rs1: int, imm: int) -> bytes:
    return struct.pack(
        "<I", ((imm & 0xFFF) << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x13
    )


def enc_cmv(rd: int, rs2: int) -> bytes:
    return struct.pack("<H", (0b1000 << 12) | (rd << 7) | (rs2 << 2) | 0b10)


# Register numbers
A1, A2, A5, S2 = 11, 12, 15, 18

CAVE_VA = 0x24CF5A
CAVE_LEN = 38
HOOK_VA = 0x2358D6
FILL_VA = 0x2358F8
HANDLER_VA = 0x23674A
MSG_VA = 0x2528B8

# (file offset, original bytes, patched bytes): the v1+v2 sites.
PATCHES = (
    # c.li a5, 1 -> c.li a5, 2, then sb to link state +0x26
    (0x2D8F2, bytes.fromhex("8547"), bytes.fromhex("8947")),
    (0x2DD2E, bytes.fromhex("8547"), bytes.fromhex("8947")),
    # sb zero, 0x26(s0) -> nop (peer-timeout clear of link state)
    (0x0F778, bytes.fromhex("23030402"), NOP4),
    # c.beqz a5, +0x78 -> c.j +0x78 (never take the "tx skip" path)
    (0x360DE, bytes.fromhex("a5cf"), bytes.fromhex("a5a8")),
    # beqz a5, +0x464 -> nop (bit 0 of slot+4 gates the send body)
    (0x3617A, bytes.fromhex("63820746"), NOP4),
    # bne s1, a5, +0x4e -> jal x0, +0x4e (skip disconnect-power case)
    (0x4CB46, bytes.fromhex("6397f404"), bytes.fromhex("6f00e004")),
    # sb zero, 1(s2) -> nop (keep descriptor TX enable for unpaired user)
    (0x35CA0, bytes.fromhex("a3000900"), NOP4),
    # sb zero, 2(s2) -> nop (keep descriptor RX enable)
    (0x35CBE, bytes.fromhex("23010900"), NOP4),
)


def build_trampoline(orig_cave: bytes) -> bytes:
    """38-byte cave at 0x24cf5a (see module docstring for the listing).

    The handler never returns here; it exits via bb_mac_tx_task's
    dispatcher epilogue, which is safe only because the descriptor
    fill loop has already completed when we branch away.
    """
    def at(n: int) -> int:
        return CAVE_VA + n

    code = b""
    code += enc_beq(A5, A1, at(0), at(8))       # fill done -> handler setup
    code += enc_jal(0, at(4), FILL_VA)          # else keep filling
    code += enc_lui(A2, (MSG_VA + 0x800) >> 12)
    code += enc_addi(A2, A2, MSG_VA - (((MSG_VA + 0x800) >> 12) << 12))
    code += enc_cmv(S2, A2)
    code += enc_jal(0, at(18), HANDLER_VA)
    assert len(code) <= CAVE_LEN, len(code)
    return code + orig_cave[len(code):]


def header_word_sum(blob: bytes) -> int:
    total = 0
    for off in range(0, HEADER_LEN, 4):
        if off == 0x08:
            continue
        total = (total + struct.unpack_from("<I", blob, off)[0]) & 0xFFFFFFFF
    return total


def spl_word_sum(blob: bytes) -> int:
    spl_len = struct.unpack_from("<I", blob, SPL_LEN_OFF)[0]
    if spl_len % 8:
        raise SystemExit(f"spl length 0x{spl_len:x} is not a multiple of 8")
    end = SPL_FILE_OFF + spl_len
    if end > len(blob):
        raise SystemExit(f"spl extends past end of image (0x{end:x} > 0x{len(blob):x})")
    total = 0
    for off in range(SPL_FILE_OFF, end, 8):
        total = (total + struct.unpack_from("<Q", blob, off)[0]) & 0xFFFFFFFFFFFFFFFF
    return total


def apply_patch(blob: bytearray) -> None:
    digest = hashlib.md5(blob).hexdigest()
    if digest != EXPECTED_MD5:
        raise SystemExit(
            f"refusing to patch: md5 {digest} != {EXPECTED_MD5} "
            "(this script's offsets are for that cx472 image only)"
        )
    stored = struct.unpack_from("<I", blob, 0x08)[0]
    if stored != header_word_sum(blob):
        raise SystemExit(
            f"header checksum 0x{stored:08x} != word sum 0x{header_word_sum(blob):08x}"
        )
    if struct.unpack_from("<Q", blob, SPL_SUM_OFF)[0] != spl_word_sum(blob):
        raise SystemExit("spl checksum mismatch; not the image this script was written against")

    for off, old, new in PATCHES:
        have = bytes(blob[off:off + len(old)])
        if have != old:
            raise SystemExit(
                f"refusing to patch at 0x{off:x}: expected {old.hex()}, found {have.hex()}"
            )
        blob[off:off + len(new)] = new

    # v3 hook: fill-loop exit branch -> trampoline.
    hook_off = va2off(HOOK_VA)
    want = bytes.fromhex("6391b702")  # bne a5, a1, +0x22
    have = bytes(blob[hook_off:hook_off + 4])
    if have != want:
        raise SystemExit(
            f"refusing to hook at 0x{hook_off:x}: expected {want.hex()}, found {have.hex()}"
        )
    blob[hook_off:hook_off + 4] = enc_jal(0, HOOK_VA, CAVE_VA)

    # Trampoline into the alignment cave (kept bytes after the code are
    # the original padding, asserted to be a zero/nop run by the caller
    # having matched the image md5 above).
    cave_off = va2off(CAVE_VA)
    orig_cave = bytes(blob[cave_off:cave_off + CAVE_LEN])
    blob[cave_off:cave_off + CAVE_LEN] = build_trampoline(orig_cave)

    # Fake CONNECT msg: 8 zero bytes, set byte5 = 2.
    msg_off = va2off(MSG_VA)
    if bytes(blob[msg_off:msg_off + 8]) != b"\x00" * 8:
        raise SystemExit(f"fake msg area at 0x{msg_off:x} is not zeroed")
    blob[msg_off + 5] = 2

    struct.pack_into("<Q", blob, SPL_SUM_OFF, spl_word_sum(blob))
    struct.pack_into("<I", blob, 0x08, header_word_sum(blob))
    if header_word_sum(blob) != struct.unpack_from("<I", blob, 0x08)[0]:
        raise SystemExit("header checksum rewrite did not stick")
    if spl_word_sum(blob) != struct.unpack_from("<Q", blob, SPL_SUM_OFF)[0]:
        raise SystemExit("spl checksum rewrite did not stick")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"usage: {Path(sys.argv[0]).name} <ar8030.img> [out.img]", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src
    blob = bytearray(src.read_bytes())
    apply_patch(blob)
    dst.write_bytes(blob)
    print(
        f"patched {dst} md5 {hashlib.md5(blob).hexdigest()} "
        f"({len(PATCHES) + 3} sites, header and spl checksums rewritten)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
