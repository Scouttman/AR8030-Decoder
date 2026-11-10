"""AR8030 demod output -> packages -> user byte stream -> AFRM chunks -> H.265 frames.

Each burst carries two packages: data symbols 0-11 hold package k, symbols 12-23
package k+1. Packages are decoded in stream order: burst n first half, burst n
second half, burst n+1 first half, ...

Package layout: bytes 0..1 length, byte 4 bits 7..5 burst counter, bytes 12..15
user TLV `01 <port+1> <len<<4 | n, LE>` when user data is present, user data from
byte 53, CRC-16/XMODEM over 0..1725 in 1726..1727.

The user bytes of the video port carry 22-byte little-endian `AFRM` chunk headers,
each followed by payload_len bytes. A frame is complete when all chunk_count chunks
with the same frame_seq arrived with a good CRC-16/CCITT-FALSE (over the payload).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass, field

import numpy as np

from .fec import CellMap, crc_ok, soft_cells, symbol_noise

USER_OFFSET = 53
AFRM_MAGIC = b"AFRM"
AFRM_HDR = struct.Struct("<IHHHHBBIHH")  # magic seq idx count len flags codec pts crc reserved
CODECS = {1: "h265", 2: "opus", 3: "ctrl", 4: "sidecar"}


# ---- packages ---------------------------------------------------------------

@dataclass
class Package:
    burst: int        # demod burst_index
    half: int         # 0 = symbols 0-11, 1 = symbols 12-23
    ok: bool          # CRC-16/XMODEM passed
    data: bytes       # 1728 bytes

    @property
    def length(self) -> int:
        return self.data[0] << 8 | self.data[1]

    @property
    def counter(self) -> int:
        return self.data[4] >> 5

    def user(self) -> tuple[int, bytes] | None:
        """(port, user bytes), or None for a package without a user TLV."""
        d = self.data
        if d[12] != 0x01:
            return None
        ulen = (d[14] | d[15] << 8) >> 4
        if USER_OFFSET + ulen > 1726:
            return None
        return d[13] - 1, bytes(d[USER_OFFSET:USER_OFFSET + ulen])


def load_symbols(path: str):
    npz = os.path.join(path, "symbols.npz")
    if not os.path.isfile(npz):
        raise SystemExit(f"no symbols.npz in {path}")
    return np.load(npz)


def _units(m: CellMap, lc: np.ndarray, half: int) -> np.ndarray:
    """Cell LLRs -> unit LLRs of one package. Missing second-half cells are erasures."""
    if half == 0:
        return lc[:, m.first]
    return np.where(m.twin != m.first, lc[:, m.twin], 0.0)


def decode_packages(path: str, ref: CellMap, first_only: bool = False) -> list[Package]:
    """Decode both packages of every burst in a demod output dir."""
    raw = load_symbols(path)
    bi = raw["burst_index"]
    B = raw["data"].shape[0]
    m = ref.for_capture(raw)
    lc = soft_cells(m.layout, raw) * np.where(m.predicted_cells(), -1.0, 1.0)
    out: list[list[Package]] = [[] for _ in range(B)]
    noise = None
    for h in (0,) if first_only else (0, 1):
        info, _ = m.decode(_units(m, lc, h))
        by = np.packbits(info, axis=1)
        ok = np.array([crc_ok(b) for b in by])
        for i in np.flatnonzero(~ok):
            noise = symbol_noise(raw) if noise is None else noise
            r = retry_erased(m, lc[i], noise[i], h)
            if r is not None:
                by[i], ok[i] = r, True
        for i in range(B):
            out[i].append(Package(int(bi[i]), h, bool(ok[i]), by[i].tobytes()))
    return [p for ps in out for p in ps]


RETRY_ERASE = 3


def retry_erased(m: CellMap, lc_row, noise_row, half: int):
    """A package that failed CRC, decoded again with its noisiest data symbols erased:
    the worst one, then the worst two, ... up to RETRY_ERASE. Packed bytes or None."""
    syms = np.arange(12) + 12 * half
    order = syms[np.argsort(noise_row[syms])[::-1]]
    for n in range(1, RETRY_ERASE + 1):
        lc = np.where(np.isin(m.layout.sym, order[:n]), 0.0, lc_row)[None, :]
        info, _ = m.decode(_units(m, lc, half))
        by = np.packbits(info, axis=1)[0]
        if crc_ok(by):
            return by
    return None


# ---- user stream ------------------------------------------------------------

@dataclass
class Segment:
    """User bytes from consecutive good packages; a gap precedes every segment but the first."""
    data: bytearray = field(default_factory=bytearray)
    packages: int = 0


def user_stream(pkgs: list[Package], port: int) -> tuple[list[Segment], dict]:
    """Join the user bytes of one port. A failed package, or a missing burst, breaks the stream."""
    segs = [Segment()]
    st = {"packages": len(pkgs), "crc_fail": 0, "idle": 0, "other_port": 0, "missing_bursts": 0, "user_bytes": 0}
    prev = None
    for p in pkgs:
        if prev is not None and p.half == 0 and p.burst != prev + 1:
            st["missing_bursts"] += p.burst - prev - 1
            if segs[-1].data:
                segs.append(Segment())
        if p.half == 0:
            prev = p.burst
        if not p.ok:
            st["crc_fail"] += 1
            if segs[-1].data:
                segs.append(Segment())
            continue
        u = p.user()
        if u is None:
            st["idle"] += 1
            continue
        if u[0] != port:
            st["other_port"] += 1
            continue
        segs[-1].data += u[1]
        segs[-1].packages += 1
        st["user_bytes"] += len(u[1])
    return [s for s in segs if s.data], st


# ---- AFRM chunks ------------------------------------------------------------

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass
class Chunk:
    seq: int
    idx: int
    count: int
    flags: int
    codec: int
    pts: int
    ok: bool
    payload: bytes


def parse_chunks(buf: bytes) -> tuple[list[Chunk], dict]:
    """Walk one segment: sync on AFRM, read header + payload, check CRC; resync on garbage."""
    chunks, st = [], {"chunks": 0, "crc_bad": 0, "resync_bytes": 0, "truncated": 0}
    i = buf.find(AFRM_MAGIC)
    st["resync_bytes"] += len(buf) if i < 0 else i
    while 0 <= i:
        if i + AFRM_HDR.size > len(buf):
            st["truncated"] += 1
            break
        _, seq, idx, cnt, ln, fl, codec, pts, crc, _ = AFRM_HDR.unpack_from(buf, i)
        end = i + AFRM_HDR.size + ln
        if cnt == 0 or idx >= cnt or codec not in CODECS:
            j = buf.find(AFRM_MAGIC, i + 1)
            st["resync_bytes"] += (len(buf) if j < 0 else j) - i
            i = j
            continue
        if end > len(buf):
            st["truncated"] += 1
            break
        pl = bytes(buf[i + AFRM_HDR.size:end])
        ok = crc16_ccitt(pl) == crc
        st["chunks"] += 1
        st["crc_bad"] += not ok
        chunks.append(Chunk(seq, idx, cnt, fl, codec, pts, ok, pl))
        i = end
        if i < len(buf) and buf[i:i + 4] != AFRM_MAGIC:
            j = buf.find(AFRM_MAGIC, i)
            st["resync_bytes"] += (len(buf) if j < 0 else j) - i
            i = j
    return chunks, st


def frames(chunks: list[Chunk], codec: int = 1):
    """Yield (seq, pts, flags, payload) for every complete frame of one codec, in arrival order."""
    cur, have = None, {}
    for c in chunks:
        if c.codec != codec:
            continue
        if cur is not None and (c.seq, c.pts) != cur[:2]:
            have = {}
        cur = (c.seq, c.pts, c.count, c.flags)
        if c.ok:
            have[c.idx] = c.payload
        if len(have) == c.count:
            yield c.seq, c.pts, c.flags, b"".join(have[k] for k in range(c.count))
            cur, have = None, {}


# ---- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symdirs", nargs="+", help="demod output dirs (symbols.npz), in time order")
    ap.add_argument("--port", type=int, default=2, help="user data port (video = 2)")
    ap.add_argument("--first-only", action="store_true", help="decode symbols 0-11 only")
    ap.add_argument("--stream", help="write the joined user bytes here (segments back to back)")
    ap.add_argument("--h265", help="write complete H.265 frames here (Annex-B)")
    a = ap.parse_args()

    ref = CellMap()
    allsegs, total = [], {}
    for name in a.symdirs:
        pkgs = decode_packages(name, ref, a.first_only)
        segs, st = user_stream(pkgs, a.port)
        for h in (0, 1):
            hp = [p for p in pkgs if p.half == h]
            if hp:
                print(f"{name}: half {h}: CRC {sum(p.ok for p in hp)}/{len(hp)}")
        print(f"{name}: {st}; {len(segs)} segments, longest {max((len(s.data) for s in segs), default=0)} B")
        allsegs += segs
        for k, v in st.items():
            total[k] = total.get(k, 0) + v

    if a.stream:
        with open(a.stream, "wb") as f:
            for s in allsegs:
                f.write(s.data)

    chunks, cst = [], {}
    for s in allsegs:
        c, st = parse_chunks(bytes(s.data))
        chunks += c
        for k, v in st.items():
            cst[k] = cst.get(k, 0) + v
    by_codec = {}
    for c in chunks:
        by_codec[CODECS[c.codec]] = by_codec.get(CODECS[c.codec], 0) + 1
    print(f"chunks: {cst}, by codec {by_codec}")

    nf = nb = idr = 0
    out = open(a.h265, "wb") if a.h265 else None
    for seq, pts, fl, data in frames(chunks):
        nf += 1
        nb += len(data)
        idr += bool(fl & 1)
        if out:
            out.write(data)
    if out:
        out.close()
    print(f"complete H.265 frames: {nf} ({idr} IDR), {nb} bytes")
    for c in chunks:
        if c.codec == 3 and c.ok:
            print("ctrl:", c.payload.decode(errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
