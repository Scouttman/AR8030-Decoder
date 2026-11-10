"""AR8030 payload: cells -> coded bits -> LDPC -> package bytes.

Transmit chain (see docs/AR8030_Airformat.md, layers 3-4):
1. Randomizer: info bits (8 codewords x 1728) XOR one continuous 1 + x^14 + x^15
   stream that starts with RANDOMIZER_HEAD, the same in every burst.
2. LDPC IEEE 802.16e (2304, 1728) rate 3/4 B, systematic; every parity bit inverted.
3. Interleaver: the measured cell map in data/cellmap.npz.
4. Cell signs: I/Q sign bits XOR an m-sequence of 1 + x^2 + x^11 indexed by
   (tone - symbol), plus a per-(symbol, I/Q) constant.

CRC: CRC-16/XMODEM over info bytes 0..1725, stored big-endian in 1726..1727.
"""
from __future__ import annotations

import os

import numpy as np

# ---- LDPC 802.16e (2304, 1728) rate 3/4 B ---------------------------------

Z = 96
N, K = 2304, 1728
M = N - K
NCW = 8  # codewords per package

# IEEE 802.16e-2005 rate 3/4 B base matrix (6 x 24, -1 = zero block).
BASE_34B = np.array([
    [-1, 81, -1, 28, -1, -1, 14, 25, 17, -1, -1, 85, 29, 52, 78, 95, 22, 92, 0, 0, -1, -1, -1, -1],
    [42, -1, 14, 68, 32, -1, -1, -1, -1, 70, 43, 11, 36, 40, 33, 57, 38, 24, -1, 0, 0, -1, -1, -1],
    [-1, -1, 20, -1, -1, 63, 39, -1, 70, 67, -1, 38, 4, 72, 47, 29, 60, 5, 80, -1, 0, 0, -1, -1],
    [64, 2, -1, -1, 63, -1, -1, 3, 51, -1, 81, 15, 94, 9, 85, 36, 14, 19, -1, -1, -1, 0, 0, -1],
    [-1, 53, 60, 80, -1, 26, 75, -1, -1, -1, -1, 86, 77, 1, 3, 72, 60, 25, -1, -1, -1, -1, 0, 0],
    [77, -1, -1, -1, 15, 28, -1, 35, -1, 72, 30, 68, 85, 84, 26, 64, 11, 89, 0, -1, -1, -1, -1, 0],
])


def parity_check() -> np.ndarray:
    """H (576 x 2304) over GF(2)."""
    mb, nb = BASE_34B.shape
    H = np.zeros((mb * Z, nb * Z), np.uint8)
    r = np.arange(Z)
    for i in range(mb):
        for j in range(nb):
            s = BASE_34B[i, j]
            if s >= 0:
                H[i * Z + r, j * Z + (r + s) % Z] = 1
    return H


def generator_parity(H: np.ndarray) -> np.ndarray:
    """P (576 x 1728): parity = P @ info mod 2, for H = [Hi | Hp]."""
    A = np.concatenate([H[:, K:], H[:, :K]], axis=1).astype(np.uint8)  # [Hp | Hi]
    for r in range(M):
        p = r + np.flatnonzero(A[r:, r])[0]
        A[[r, p]] = A[[p, r]]
        rows = np.flatnonzero(A[:, r])
        A[rows[rows != r]] ^= A[r]
    return A[:, M:]


def encode(info_bits: np.ndarray, H: np.ndarray) -> np.ndarray:
    """info_bits (13824,) -> (8, 2304) coded words, info first."""
    P = generator_parity(H)
    u = np.asarray(info_bits, np.uint8).reshape(NCW, K)
    par = (u.astype(np.int32) @ P.T.astype(np.int32)) % 2
    return np.concatenate([u, par.astype(np.uint8)], axis=1)


def minsum(H: np.ndarray, llr: np.ndarray, iters: int = 50, alpha: float = 0.75):
    """Normalized min-sum. llr (B, N): >0 means bit 1. Returns (bits, syndrome_ok)."""
    rows = [np.flatnonzero(H[r]) for r in range(H.shape[0])]
    dmax = max(len(r) for r in rows)
    idx = np.zeros((H.shape[0], dmax), np.int64)
    msk = np.zeros((H.shape[0], dmax), bool)
    for r, cols in enumerate(rows):
        idx[r, :len(cols)] = cols
        msk[r, :len(cols)] = True
    L0 = -llr.astype(np.float64)  # inside: >0 means bit 0
    R = np.zeros((llr.shape[0], H.shape[0], dmax))
    post = L0.copy()
    for _ in range(iters):
        Q = np.where(msk, post[:, idx] - R, 1e9)
        sg = np.where(Q < 0, -1.0, 1.0)
        mag = np.abs(Q)
        sgp = np.prod(sg, axis=2, keepdims=True)
        o = np.sort(mag, axis=2)
        m1, m2 = o[:, :, :1], o[:, :, 1:2]
        R = np.where(msk, alpha * sgp * sg * np.where(mag == m1, m2, m1), 0.0)
        post = L0.copy()
        np.add.at(post, (slice(None), idx[msk]), R[:, msk])
        ok = (((post < 0).astype(np.int32) @ H.T.astype(np.int32)) % 2 == 0).all(1)
        if ok.all():
            break
    return (post < 0).astype(np.uint8), ok


# ---- scrambling and CRC ----------------------------------------------------

RANDOMIZER_HEAD = "000000111111011"  # stream bits 0..14 (codeword 0, info bit 0 first)
CELL_O = 20
CELL_CONST_I = "111101000000101111110100"  # data symbols 0..23
CELL_CONST_Q = "101111110100000010111111"


def randomizer_stream(n: int = NCW * K) -> np.ndarray:
    s = [int(c) for c in RANDOMIZER_HEAD]
    while len(s) < n:
        s.append(s[-14] ^ s[-15])
    return np.array(s[:n], np.uint8)


def cell_prbs(n: int = 2047 + 800) -> np.ndarray:
    s = [1] * 11
    while len(s) < n:
        s.append(s[-2] ^ s[-11])
    return np.array(s, np.uint8)


def column_flips(layout: "Layout") -> np.ndarray:
    """(columns,) XOR on each cell bit after coding (sign planes only)."""
    sym = layout.sym.astype(int)
    w = cell_prbs()[CELL_O + layout.sc.astype(int) + 216 - sym + 100]
    ci = np.array([int(c) for c in CELL_CONST_I], np.uint8)
    cq = np.array([int(c) for c in CELL_CONST_Q], np.uint8)
    const = np.where(layout.plane == 0, ci[sym], cq[sym])
    return np.where(layout.plane < 2, w ^ const, 0).astype(np.uint8)


def crc16_xmodem(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def crc_ok(package: np.ndarray) -> bool:
    """package (1728,) bytes."""
    return crc16_xmodem(bytes(package[:1726])) == (int(package[1726]) << 8 | int(package[1727]))


# ---- cell layout and map ---------------------------------------------------

PLANES = ("I_sign", "Q_sign", "I_mag", "Q_mag")
TWIN = 12  # package k+1 sits 12 symbols after package k
MAP_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cellmap.npz")


class Layout:
    """Cell columns: data symbol, then tone, then the four 16-QAM bit planes."""

    def __init__(self, sym, sc, plane):
        self.sym = np.asarray(sym, np.int16)
        self.sc = np.asarray(sc, np.int16)
        self.plane = np.asarray(plane, np.uint8)
        self.n = int(self.sym.size)

    @classmethod
    def from_capture(cls, cap, nsym: int) -> "Layout":
        """Every data tone of a demod output except the empty mid-pilot bins."""
        dm, k, offs = cap["data_mask"], cap["subcarrier"], cap["pilot_offsets"]
        syms, scs, planes = [], [], []
        for s in range(nsym):
            off = int(offs[2 + s])  # two FCH symbols first
            for idx in np.where(dm[2 + s])[0]:
                sc = int(k[idx])
                if (sc - off) % 18 == 9:
                    continue
                for p in range(4):
                    syms.append(s)
                    scs.append(sc)
                    planes.append(p)
        return cls(syms, scs, planes)


def twin_plane(sym: int, plane: int) -> int:
    """Plane of the +12 copy: I and Q swap when the rail rule flips."""
    def a_is_q(s):
        return (s % 8) in (0, 1, 2, 7)
    return plane if a_is_q(sym) == a_is_q(sym + TWIN) else plane ^ 1


class CellMap:
    """Where every coded bit of a package sits.

    col_unit (8, 2304): unit of each (codeword, column), -1 where no cell is known.
    A unit is a cell column of the first package (symbols 0-11): `first`. `twin` is
    the matching column of the second package (symbols 12-23), or `first` if absent.
    """

    def __init__(self, path: str = MAP_NPZ):
        z = np.load(path)
        self.col_unit = z["col_unit"].astype(np.int64)
        self.first = z["first"].astype(np.int64)
        self.twin = z["twin"].astype(np.int64)
        self.layout = Layout(z["sym"], z["sc"], z["plane"])
        self.H = parity_check()

    def for_capture(self, raw) -> "CellMap":
        """Copy whose twins cover data symbols 22-23 (present when the demod kept 24 data symbols)."""
        out = CellMap.__new__(CellMap)
        out.col_unit, out.first, out.H = self.col_unit, self.first, self.H
        out.twin, out.layout = self.twin.copy(), self.layout
        nsym = raw["data"].shape[1]
        if nsym < 24:
            return out
        L = Layout.from_capture(raw, nsym)
        n = self.layout.n
        if not ((L.sym[:n] == self.layout.sym).all() and (L.sc[:n] == self.layout.sc).all()):
            raise SystemExit("capture layout does not extend the reference layout")
        col = {(int(s), int(c), int(p)): i for i, (s, c, p) in enumerate(zip(L.sym, L.sc, L.plane))}
        for u in np.flatnonzero(out.twin == out.first):
            f = out.first[u]
            s = int(L.sym[f])
            j = col.get((s + TWIN, int(L.sc[f]), twin_plane(s, int(L.plane[f]))))
            if j is not None:
                out.twin[u] = j
        out.layout = L
        return out

    def predicted_cells(self) -> np.ndarray:
        """Cell bits (columns,) of an all-zero package: the scrambling reference."""
        c = encode(np.zeros(NCW * K, np.uint8) ^ randomizer_stream(), self.H)
        c[:, K:] ^= 1  # parity inverted
        cb = np.zeros(self.layout.n, np.uint8)
        cov = np.zeros(self.layout.n, bool)
        for w in range(NCW):
            u = self.col_unit[w]
            k = u >= 0
            for cols in (self.first[u[k]], self.twin[u[k]]):
                cb[cols] = c[w, k]
                cov[cols] = True
        return (cb ^ column_flips(self.layout)).astype(bool) & cov

    def decode(self, lu: np.ndarray, iters: int = 50):
        """lu (bursts, units) LLRs -> (info (bursts, 13824), ok (bursts, 8)). Unmapped columns are erasures."""
        B = lu.shape[0]
        info = np.zeros((B, NCW, K), np.uint8)
        ok = np.zeros((B, NCW), bool)
        for c in range(NCW):
            cu = self.col_unit[c]
            k = cu >= 0
            llr = np.zeros((B, N))
            llr[:, k] = lu[:, cu[k]]
            bits, okc = minsum(self.H, llr, iters=iters)
            info[:, c], ok[:, c] = bits[:, :K], okc
        return info.reshape(B, -1), ok


# ---- soft cells ------------------------------------------------------------

def _live_cells(raw, nsym: int) -> np.ndarray:
    dm = raw["data_mask"][-nsym:].astype(bool)
    k = raw["subcarrier"]
    empty = np.array([((k - int(o)) % 18) == 9 for o in raw["pilot_offsets"][-nsym:]])
    return dm & ~empty


def _scaled_data(raw):
    z = raw["data"]
    live = _live_cells(raw, z.shape[1])
    outer = np.abs(raw["data_i"][:, live]) == 3
    return z / (np.median(np.abs(z.real[:, live])[outer]) / 3), live


def _symbol_var(z, live) -> np.ndarray:
    q = lambda x: np.clip(2 * np.floor(x / 2) + 1, -3, 3)  # noqa: E731
    err = np.abs(z - (q(z.real) + 1j * q(z.imag))) ** 2
    var = np.array([[err[b, s, live[s]].mean() for s in range(z.shape[1])] for b in range(z.shape[0])])
    return np.maximum(var / 2, 0.02)  # per real dimension


def symbol_noise(raw) -> np.ndarray:
    """(bursts, symbols) noise per data symbol in dB above that symbol's median over bursts."""
    db = 10 * np.log10(_symbol_var(*_scaled_data(raw)))
    return db - np.median(db, axis=0, keepdims=True)


def soft_cells(layout: Layout, raw) -> np.ndarray:
    """(bursts, columns) LLR that the cell's plane bit is 1, from soft I/Q."""
    z, live = _scaled_data(raw)
    var = _symbol_var(z, live)
    index = {int(sc): i for i, sc in enumerate(raw["subcarrier"])}
    t = np.fromiter((index[int(sc)] for sc in layout.sc), np.int32, count=layout.n)
    x = z[:, layout.sym, t]
    re, im = x.real, x.imag
    p = layout.plane
    val = np.where(p == 0, re, np.where(p == 1, im, np.where(p == 2, np.abs(re) - 2, np.abs(im) - 2)))
    return 2 * val / var[:, layout.sym]
