"""Demodulate the AR8030 downlink from an IQ recording to equalized symbols.

The recording is interleaved little-endian int16 I/Q (cs16) at 40.0016 MS/s. The output
(symbols.npz) holds equalized constellation points and hard decisions per burst; stream.py
decodes the packages from it. See docs/AR8030_Airformat.md for the frame layout.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import firwin, resample_poly

FS = 40.0016e6
NFFT = 512
CP = 32
SYM = NFFT + CP
DOWN = 3657
RATE = FS * NFFT / DOWN
DF = RATE / NFFT
PILOT_SPACING = 18
K = np.arange(-216, 218)
KIDX = K % NFFT
KBIN = np.fft.fftfreq(NFFT, 1 / NFFT).astype(int)
BACK = 8
QAM16 = np.array([-3, -1, 1, 3]) / np.sqrt(10)

# Transmitted pilot values for the comb 0,9,12,3,15,6 (repeating), symbols 0..1 FCH
# and 2..23 data. Each row: axis ('r' = ±1, 'i' = ±j), then one sign per comb tone
# (K - offset) % 18 == 0, K != 0, in increasing K ('0' = edge bin -216, unused).
# BPSK, identical in every burst and capture. The pattern is not rotation symmetric,
# so it fixes all four quarter turns of each symbol and the capture-wide 180 degrees
# that the header channel estimate leaves open.
KNOWN_PILOT_OFFSETS = (0, 9, 12, 3, 15, 6)
KNOWN_PILOTS = (
    "r0--+---+-+++---+-++-+-+-",  # 0
    "r++--+-+-++-++--+-++--+++",  # 1
    "i+-++---+-+-++---+-+++-++",  # 2
    "i+--+---+-++++---+-++-+-+",  # 3
    "i+-+-+-----+-+++-+------+",  # 4
    "r-++----+---+--+-+--+----",  # 5
    "r0--+-+++++-+-+-+++-+-+--",  # 6
    "r++++--+---+-++---+-+----",  # 7
    "r++-++--+++---+--+++-++++",  # 8
    "i+--+-+++++-++-+-+++-+-+-",  # 9
    "i++-+---++--++-++++-++---",  # 10
    "i-+-+--+-+++++++-+--+-+++",  # 11
    "i0+---+-++--++-+--++++---",  # 12
    "r-+--+-++-----+-++---++--",  # 13
    "r+-+---++++--+----++-----",  # 14
    "r--+++-+--++---+-++----++",  # 15
    "r+----+---++++++---+---++",  # 16
    "i++--++-+-+--++++--+-+-+-",  # 17
    "i0+--+---+-+++---+-++-+-+",  # 18
    "i+++--+-+-++-++--+-++--++",  # 19
    "i++-++---+-+-++---+-+++-+",  # 20
    "r+-++-+++-+----+++-+--+-+",  # 21
    "r--+-+-+++++-+---+-++++++",  # 22
    "r+-++----+---+--+-+--+---",  # 23
    # 24-25: data symbols 22-23, the second package's last two symbols.
    "r0+--+-+++++-+-+-+++-+-+-",  # 24
    "i-++++--+---+-++---+-+---",  # 25
)


def resnap(data: np.ndarray, pilot_mask: np.ndarray, iters: int = 8):
    """Re-snap each (burst, symbol) by quarter-turns against the per-pilot median.

    Only used with learned pilots: their mean reference is dragged when a share of
    bursts slips 90 degrees. Returns (data, k), data rotated by (-j)**k."""
    z = np.array(data, copy=True)
    k_total = np.zeros(z.shape[:2], np.int8)
    for _ in range(iters):
        moved = 0
        for d in range(z.shape[1]):
            p = z[:, d, pilot_mask[d]]
            if p.shape[1] == 0:
                continue
            ref = np.median(p.real, axis=0) + 1j * np.median(p.imag, axis=0)
            sc = np.stack([np.real(np.sum(p * np.conj(ref) * (-1j) ** k, axis=1)) for k in range(4)], 1)
            k = sc.argmax(1)
            m = k != 0
            if np.any(m):
                z[m, d, :] *= ((-1j) ** k[m])[:, None]
                k_total[m, d] = (k_total[m, d] + k[m]) % 4
                moved += int(m.sum())
        if not moved:
            break
    return z, k_total


def known_pilots(offs, pil):
    """Known pilot values on pil[j]'s tones, as track() returns P; None if the comb is another."""
    if len(offs) > len(KNOWN_PILOTS):
        return None
    if any(o != KNOWN_PILOT_OFFSETS[j % len(KNOWN_PILOT_OFFSETS)] for j, o in enumerate(offs)):
        return None
    P = []
    for j, o in enumerate(offs):
        row = KNOWN_PILOTS[j]
        c = np.where(((K - o) % PILOT_SPACING == 0) & (K != 0))[0]
        full = np.zeros(K.size, complex)
        full[c] = np.array([{"+": 1, "-": -1, "0": 0}[ch] for ch in row[1:]]) * (1 if row[0] == "r" else 1j)
        P.append(full[pil[j]])
    return P


def pilot_agreement(Q, pil, P):
    """Per symbol, median over bursts of |<pilots, P>| / (|pilots| |P|), rotation free."""
    out = []
    for j in range(Q.shape[1]):
        v = Q[:, j, pil[j]]
        num = np.abs(v @ np.conj(P[j]))
        out.append(float(np.median(num / (np.linalg.norm(v, axis=1) * np.linalg.norm(P[j]) + 1e-12))))
    return np.array(out)


def find_bursts(raw, skip_s, min_on_s=2.0e-3):
    w = 400
    start = int(skip_s * FS)
    x = np.asarray(raw[2 * start:]).astype(np.float32)
    p = x[0::2] ** 2 + x[1::2] ** 2
    e = p[: p.size // w * w].reshape(-1, w).mean(axis=1)
    thr = np.sqrt(np.percentile(e, 10) * np.percentile(e, 90))
    d = np.diff(np.r_[0, (e > thr).astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (en - st) * w / FS > min_on_s
    return np.stack([st[keep] * w + start, en[keep] * w + start], axis=1)


# resample_poly's own default filter (Kaiser beta 5, 10*max(up, down) half
# length), designed once. scipy redesigned this ~73k-tap filter on every
# burst, which was ~30% of the demod's run time.
_AA = firwin(2 * 10 * max(NFFT, DOWN) + 1, 1.0 / max(NFFT, DOWN), window=("kaiser", 5.0)).astype(np.complex64)
_MIX = {}


def _mixer(shift_hz, n):
    """exp(-2j*pi*shift*k/FS) for k < n, complex64, cached per shift."""
    tab = _MIX.get(shift_hz)
    if tab is None or tab.size < n:
        k = np.arange(max(n, 1 << 17), dtype=np.float64)
        tab = np.exp(-2j * np.pi * ((shift_hz * k / FS) % 1.0)).astype(np.complex64)
        _MIX[shift_hz] = tab
    return tab[:n]


def to_native(raw, s0, s1, shift_hz, pad_s=40e-6):
    nsamp = raw.size // 2
    pad = int(pad_s * FS)
    a, b = max(0, s0 - pad), min(nsamp, s1 + pad)
    z = np.asarray(raw[2 * a:2 * b]).astype(np.float32).view(np.complex64)
    # Same mix as exp(-2j*pi*shift*(k+a)/FS): a cached table times one start phase.
    start = np.complex64(np.exp(-2j * np.pi * ((shift_hz * a / FS) % 1.0)))
    z = z * (_mixer(shift_hz, z.size) * start)
    return resample_poly(z, NFFT, DOWN, window=_AA).astype(np.complex64)


def lag_metric(z, lag, w):
    c = z[:-lag] * np.conj(z[lag:])
    cs = np.r_[0, np.cumsum(c)]
    mov = cs[w:] - cs[:-w]
    ps = np.r_[0, np.cumsum(np.abs(z) ** 2)]
    m = mov.size
    pa = ps[w:w + m] - ps[:m]
    pb = ps[lag + w:lag + w + m] - ps[lag:lag + m]
    return np.abs(mov) / np.sqrt(pa * pb + 1e-9), mov


# A receiver click (a few samples of wideband junk) inside the header splits the
# copy-vs-copy plateau in two, e.g. 294 + 202 samples, and the burst was dropped as
# "no header lock" although the rest of it is clean. Runs closer than this are joined; the click itself is left out of
# the frequency estimate.
LOCK_GAP = 48


def header_lock(z):
    """Start of the two-copy header and the residual frequency offset it implies."""
    met, mov = lag_metric(z, NFFT, 16)
    on = met > 0.9
    d = np.diff(np.r_[0, on.astype(int), 0])
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    if s.size == 0:
        return None
    keep = np.r_[True, s[1:] - e[:-1] > LOCK_GAP]
    s, e = s[keep], np.r_[e[:-1][keep[1:]], e[-1]]
    h = np.argmax(e - s)
    ha, hb = int(s[h]), int(e[h])
    if hb - ha < 450:
        return None
    cfo = -np.angle(np.sum(mov[ha:hb][on[ha:hb]])) / (2 * np.pi * NFFT / RATE)
    t = np.arange(z.size) / RATE
    return z * np.exp(-2j * np.pi * cfo * t).astype(np.complex64), ha, float(cfo)


# Sync check. Some receivers (seen on an MT7921) corrupt a stretch of a few tens of
# microseconds in a burst once the signal inside it is above roughly rms 50
# (cs16 counts at the lowest held gain), far below clipping: raising only the
# receiver gain brings it back at a fixed transmit power. A hit inside the
# header shows as the two copies disagreeing there. Such bursts are flagged and
# dropped, not repaired. Keep the received level low (rms ~20-45) to avoid them.
SYNC_WIN = 16
SYNC_CLEAN = 0.9


def sync_quality(z, ha):
    """Lowest copy-vs-copy coherence over the two-copy header, per SYNC_WIN samples."""
    a, b = z[ha:ha + NFFT], z[ha + NFFT:ha + 2 * NFFT]
    if ha < 0 or b.size < NFFT:
        return 0.0
    a, b = a.reshape(-1, SYNC_WIN), b.reshape(-1, SYNC_WIN)
    num = np.abs(np.sum(a * np.conj(b), axis=1))
    den = np.sqrt(np.sum(np.abs(a) ** 2, axis=1) * np.sum(np.abs(b) ** 2, axis=1)) + 1e-12
    return float(np.min(num / den))


# header_lock's plateau start lands one CP (32 samples) late on ~10% of bursts
# (2026-11-07, r3: every symbol's CP sits at -32). A +-6 search could not reach
# that and those bursts failed all 8 codewords. Search past one CP either way.
GRID_SEARCH = CP + 8


def grid_offset(z, ha, nsym):
    best = (-1.0, 0)
    for d in range(-GRID_SEARCH, GRID_SEARCH + 1):
        acc = 0.0
        for j in range(nsym):
            c = ha + 2 * NFFT + j * SYM + d
            if c < 0 or c + SYM > z.size:
                break
            acc += abs(np.vdot(z[c:c + CP], z[c + NFFT:c + NFFT + CP]))
        if acc > best[0]:
            best = (acc, d)
    return best[1]


def burst_cells(z, ha, d, nsym):
    ramp = np.exp(2j * np.pi * KBIN * BACK / NFFT)
    h = 0.5 * (np.fft.fft(z[ha + d:ha + d + NFFT]) + np.fft.fft(z[ha + d + NFFT:ha + d + 2 * NFFT]))
    rows = []
    for j in range(nsym):
        st = ha + 2 * NFFT + j * SYM + d + CP - BACK
        if st + NFFT > z.size:
            break
        rows.append(np.fft.fft(z[st:st + NFFT]) * ramp)
    return h[KIDX], np.array(rows)[:, KIDX] if rows else None


def normalise(H, Y, hp):
    """Per-burst complex gain and timing slope against the mean header, on the header subcarriers."""
    def fit(hb, ref):
        r = hb[hp] * np.conj(ref[hp])
        s = np.angle(np.sum(r[1:] * np.conj(r[:-1]))) / 2
        a = np.sum(r * np.exp(-1j * s * K[hp])) / np.sum(np.abs(ref[hp]) ** 2)
        return a, s

    nb = H.shape[0]
    ref = H[0].copy()
    for _ in range(3):
        acc = np.zeros(K.size, complex)
        for b in range(nb):
            a, s = fit(H[b], ref)
            acc += H[b] / (a * np.exp(1j * s * K))
        ref = acc / nb
    Yn = np.empty_like(Y)
    for b in range(nb):
        a, s = fit(H[b], ref)
        Yn[b] = Y[b] / (a * np.exp(1j * s * K))[None, :]
    return ref, Yn


def channel_from_header(ref, hp):
    """The header is BPSK on its subcarriers, so its square is the squared channel."""
    kh = K[hp].astype(float)
    ph = np.unwrap(np.angle(ref[hp] ** 2)) / 2
    return np.interp(K, kh, np.abs(ref[hp])) * np.exp(1j * np.interp(K, kh, ph))


def pilot_offsets(Z):
    U = Z / np.abs(Z)
    offs, scores = [], []
    for j in range(Z.shape[1]):
        sc = []
        for o in range(PILOT_SPACING):
            c = np.where(((K - o) % PILOT_SPACING == 0) & (K != 0))[0]
            rr = U[:, j, c[1:]] * np.conj(U[:, j, c[:-1]])
            sc.append(float(np.mean(np.abs(np.mean(rr, axis=0)))))
        offs.append(int(np.argmax(sc)))
        scores.append(max(sc))
    return offs, scores


def track(Z, pil, iters=5):
    """Phase, slope and gain per (burst, symbol) against pilot values learned across bursts."""
    nb, ns, _ = Z.shape
    U = Z / np.abs(Z)
    P = [np.mean(U[:, j, pil[j]], axis=0) for j in range(ns)]
    Q = np.empty_like(Z)
    for _ in range(iters):
        for j in range(ns):
            c = np.where(pil[j])[0]
            kp = K[c].astype(float)
            r = Z[:, j, c] * np.conj(P[j])[None, :]
            s = np.angle(np.sum(r[:, 1:] * np.conj(r[:, :-1]), axis=1)) / PILOT_SPACING
            r2 = r * np.exp(-1j * s[:, None] * kp[None, :])
            th = np.angle(np.sum(r2, axis=1))
            g = np.sum(np.abs(r2), axis=1) / np.sum(np.abs(P[j]) ** 2)
            Q[:, j, :] = Z[:, j, :] * np.exp(-1j * (th[:, None] + s[:, None] * K[None, :])) / g[:, None]
            P[j] = np.mean(Q[:, j, c], axis=0)
    return Q, P


def lock_slope(Q, mask, power, slopes=np.linspace(-0.03, 0.03, 61)):
    """Residual timing slope and power-law phase. Phase is only unique mod 2π/power."""
    out = np.empty_like(Q)
    for j in range(Q.shape[1]):
        m = mask[j]
        kk = K[m].astype(float)
        v = Q[:, j, m]
        up = (v / np.abs(v)) ** power
        sc = np.abs(up @ np.exp(-1j * power * slopes[:, None] * kk[None, :]).T)
        s = slopes[np.argmax(sc, axis=1)]
        w = Q[:, j, :] * np.exp(-1j * s[:, None] * K[None, :])
        wm = w[:, m]
        th = np.angle(np.mean((wm / np.abs(wm)) ** power, axis=1)) / power
        if power == 4:
            th = th + np.pi / 4
        w = w * np.exp(-1j * th[:, None])
        out[:, j, :] = w / np.sqrt(np.mean(np.abs(w[:, m]) ** 2) + 1e-12)
    return out


def snap_fold(Q, pil, P, fold):
    """Pick the unique k·2π/fold rotation that best matches learned pilots P."""
    step = 2 * np.pi / fold
    out = np.empty_like(Q)
    nflip = 0
    for j in range(Q.shape[1]):
        c = np.where(pil[j])[0]
        r0 = Q[:, j, c] * np.conj(P[j])[None, :]
        ks = []
        for b in range(Q.shape[0]):
            scores = [float(np.real(np.sum(r0[b] * np.exp(-1j * k * step)))) for k in range(fold)]
            k = int(np.argmax(scores))
            ks.append(k)
            if k:
                nflip += 1
        th = np.array(ks, dtype=float) * step
        out[:, j, :] = Q[:, j, :] * np.exp(-1j * th[:, None])
    return out, nflip


def detect_modulation(pts):
    """A quarter of 16-QAM points sit on the inner ring at 0.45 of rms; QPSK has none there."""
    p = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    inner = float(np.mean(np.abs(p) < 0.65))
    return ("qpsk" if inner < 0.17 else "16qam"), inner


def levels(x, mod):
    if mod == "qpsk":
        return np.where(x.real > 0, 1, -1).astype(np.int8), np.where(x.imag > 0, 1, -1).astype(np.int8)
    lv = np.array([-3, -1, 1, 3], dtype=np.int8)
    i = lv[np.argmin(np.abs(x.real[..., None] - QAM16), axis=-1)]
    q = lv[np.argmin(np.abs(x.imag[..., None] - QAM16), axis=-1)]
    return i, q


def ideal(i, q, mod):
    return (i + 1j * q) / (np.sqrt(2) if mod == "qpsk" else np.sqrt(10))


def evm_db(x, ref):
    return float(20 * np.log10(np.sqrt(np.mean(np.abs(x - ref) ** 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", help="cs16 file at 40.0016 MS/s")
    ap.add_argument("--lo-mhz", type=float, default=5780.0, help="tuned center of the recording")
    ap.add_argument("--grid-mhz", type=float, default=5770.0196,
                    help="RF frequency of subcarrier 0 (residual is corrected per burst, within +-5 kHz)")
    ap.add_argument("--skip-s", type=float, default=0.1, help="skip the recorder's 20 MS/s start-up")
    ap.add_argument("--first", type=int, default=20, help="first burst to use")
    ap.add_argument("--bursts", type=int, default=100000,
                    help="bursts to try (default all; the old 400 left over half of a 2 min capture unread)")
    ap.add_argument("--fch-symbols", type=int, default=2)
    ap.add_argument("--symbols", type=int, default=26,
                    help="symbols kept per burst: 2 FCH + 24 data (data 22-23 finish the second package)")
    ap.add_argument("--keep", type=int, default=0, help="bursts written to symbols.npz, 0 = all (all are used for estimates)")
    ap.add_argument("--pilot-cycle", default=None,
                    help="comma-separated pilot offsets, repeated across the burst. Overrides the comb picker")
    ap.add_argument("--workers", type=int, default=4,
                    help="threads for the per-burst demod. Above 1 loads the recording into memory")
    ap.add_argument("--out", default="ar8030-demod", help="output directory")
    ap.add_argument("--no-resnap", dest="resnap", action="store_false",
                    help="keep snap_fold's rotations (no median-pilot re-snap of data symbols)")
    ap.add_argument("--learned-pilots", action="store_true",
                    help="snap symbol rotations to pilots learned from this capture (the old behaviour) "
                         "instead of the known transmitted pilot table")
    ap.add_argument("--sync-min", type=float, default=SYNC_CLEAN,
                    help="drop bursts whose two header copies disagree anywhere: lowest copy-vs-copy "
                         "coherence per %d samples below this (default %.2f; 0 keeps all)" % (SYNC_WIN, SYNC_CLEAN))
    a = ap.parse_args()

    # One in-memory copy so the worker threads slice RAM instead of a memmap.
    if a.workers > 1:
        raw = np.fromfile(a.recording, dtype=np.int16)
    else:
        raw = np.memmap(a.recording, dtype=np.int16, mode="r")
    shift = (a.grid_mhz - a.lo_mhz) * 1e6
    B = find_bursts(raw, a.skip_s)
    print(f"bursts found {len(B)}  native rate {RATE / 1e6:.5f} MS/s  subcarrier {DF / 1e3:.3f} kHz")

    def demod_burst(i):
        z = to_native(raw, *B[i], shift)
        r = header_lock(z)
        if r is None:
            return "nolock"
        z, ha, cfo = r
        q = sync_quality(z, ha)
        if q < a.sync_min:
            return "badsync"
        d = grid_offset(z, ha, a.symbols)
        h, y = burst_cells(z, ha, d, a.symbols)
        if y is None or len(y) < a.symbols:
            return "short"
        return h, y, (i, int(B[i][0]), ha, d, cfo, q)

    span = range(a.first, min(len(B), a.first + a.bursts))
    if a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(demod_burst, span))
    else:
        results = [demod_burst(i) for i in span]
    Hs, Ys, meta = [], [], []
    drops = {"nolock": 0, "badsync": 0, "short": 0}
    for item in results:
        if isinstance(item, str):
            drops[item] += 1
            continue
        h, y, meta_i = item
        Hs.append(h)
        Ys.append(y)
        meta.append(meta_i)
    if len(Hs) < 20:
        sys.exit(f"only {len(Hs)} bursts locked; check --lo-mhz/--grid-mhz")
    H, Y = np.array(Hs), np.array(Ys)
    nb, ns, _ = Y.shape
    cfo = np.array([m[4] for m in meta])
    print(f"dropped: no header lock {drops['nolock']}, bad sync {drops['badsync']} "
          f"(copies disagree, coherence < {a.sync_min:.2f}), too short {drops['short']}")
    print(f"locked {nb} bursts  residual offset {np.median(cfo):+.0f} Hz (sd {np.std(cfo):.0f})")

    pw = np.mean(np.abs(H) ** 2, axis=0)
    parity = 0 if pw[K % 2 == 0].mean() > pw[K % 2 == 1].mean() else 1
    hp = (K % 2 == parity)
    empty_db = 10 * np.log10(pw[~hp].mean() / pw[hp].mean())
    print(f"header on {'even' if parity == 0 else 'odd'} subcarriers, other parity {empty_db:.1f} dB")

    ref, Yn = normalise(H, Y, hp)
    C = channel_from_header(ref, hp)
    Z = Yn / C[None, None, :]
    offs, scores = pilot_offsets(Z)
    if a.pilot_cycle:
        cyc = [int(x) for x in a.pilot_cycle.split(",") if x != ""]
        if not cyc:
            sys.exit("--pilot-cycle needs at least one offset")
        offs = [cyc[j % len(cyc)] for j in range(ns)]
        print(f"pilot comb forced {offs}")
    else:
        print(f"pilot comb offsets {offs}")
    used = np.mean(np.abs(Z), axis=(0, 1)) > 0.2 * np.median(np.abs(Z))
    pil = np.array([((K - o) % PILOT_SPACING == 0) & (K != 0) & used for o in offs])
    dm = ~pil & (K != 0)[None, :] & used[None, :]
    # Data symbols leave the bin halfway between pilots empty
    # ((sc - offset) % 18 == 9; 24 per symbol, amplitude ~0.2 of a live
    # cell). Counting them as data put noise into lock_slope's 4th-power
    # phase and RMS gain (which sets the inner/outer decision) and into
    # the EVM. FCH rows keep their 385-cell mask; the FCH tools index it.
    nf = a.fch_symbols
    for j in range(nf, ns):
        dm[j] &= (K - offs[j]) % PILOT_SPACING != PILOT_SPACING // 2

    Q, P = track(Z, pil)
    # Snap reference: the known transmitted pilots when the comb is the known one.
    # The learned P is a mean over this capture's bursts, so when most bursts of a
    # symbol slip 90 degrees, P slips with them and the whole symbol comes out
    # rotated. The known values have no such vote.
    Pk = None if a.learned_pilots else known_pilots(offs, pil)
    phase_from = "learned_pilots"
    if Pk is not None:
        agree = pilot_agreement(Q, pil, Pk)
        if np.median(agree) < 0.5:
            print(f"known pilots do not fit this capture (median agreement {np.median(agree):.2f}); "
                  "using learned pilots")
            Pk = None
        else:
            print(f"known pilots fit, agreement by symbol min {agree.min():.2f} median {np.median(agree):.2f}")
            phase_from = "known_pilots"
    Pref = P if Pk is None else Pk
    fch, nfch = snap_fold(lock_slope(Q[:, :nf], dm[:nf], 2), pil[:nf], Pref[:nf], 2)
    data, ndat = snap_fold(lock_slope(Q[:, nf:], dm[nf:], 4), pil[nf:], Pref[nf:], 4)
    print(f"pilot snap ({phase_from})  FCH {nfch} sign flips / {fch.shape[0]*nf}   "
          f"data {ndat} ×90° / {data.shape[0]*(Q.shape[1]-nf)}")
    # snap_fold's reference P is a mean over all bursts. On data symbols
    # 3, 7, 11, 15, 19 a share of bursts slips -90 degrees and drags P
    # between the two, so the snap keeps choosing wrong. Re-learn P from the
    # snapped bursts (median) and re-snap. ar8030_cell_quality.py.
    # The median re-snap only repairs a learned reference; against the known
    # pilots it could only move bursts away from the transmitted values.
    if a.resnap and Pk is None:
        data, resnap_k = resnap(data, pil[nf:])
        moved = (resnap_k != 0)
        print(f"pilot re-snap  {int(moved.sum())} (burst,symbol) re-rotated, by symbol {moved.sum(0).tolist()}")
    else:
        resnap_k = np.zeros(data.shape[:2], np.int8)

    fpts = np.concatenate([fch[:, j, dm[j]].ravel() for j in range(nf)])
    dpts = np.concatenate([data[:, j - nf, dm[j]].ravel() for j in range(nf, ns)])
    mod, ratio = detect_modulation(dpts)
    li, lq = levels(dpts, mod)
    fch_evm = evm_db(fpts, np.sign(fpts.real))
    qam_evm = evm_db(dpts, ideal(li, lq, mod))
    label = "QPSK" if mod == "qpsk" else "16-QAM"
    print(f"data modulation {label} (inner-ring share {ratio:.2f})")
    print(f"FCH BPSK EVM {fch_evm:.1f} dB   data {label} EVM {qam_evm:.1f} dB")

    os.makedirs(a.out, exist_ok=True)
    di, dq = levels(data, mod)
    n = nb if a.keep <= 0 else min(a.keep, nb)
    np.savez_compressed(
        os.path.join(a.out, "symbols.npz"),
        subcarrier=K, header_ref=ref, channel=C, pilot_offsets=np.array(offs),
        pilot_mask=pil, data_mask=dm,
        fch=fch[:n].astype(np.complex64), fch_bits=(fch[:n].real > 0).astype(np.uint8),
        data=data[:n].astype(np.complex64), data_i=di[:n], data_q=dq[:n],
        burst_index=np.array([m[0] for m in meta])[:n], sync_quality=np.array([m[5] for m in meta])[:n], burst_sample=np.array([m[1] for m in meta])[:n],
        residual_hz=cfo[:n], resnap_k=resnap_k[:n],
    )
    summary = dict(
        recording=os.path.abspath(a.recording), lo_mhz=a.lo_mhz, grid_mhz=a.grid_mhz,
        bursts_found=int(len(B)), bursts_locked=int(nb), bursts_saved=int(n), native_rate_hz=RATE, fft=NFFT, cp=CP,
        subcarrier_hz=DF, header_parity="even" if parity == 0 else "odd",
        pilot_spacing=PILOT_SPACING, pilot_offsets=offs, residual_hz_median=float(np.median(cfo)),
        fch_symbols=nf, data_symbols=ns - nf, data_modulation=mod, data_inner_share=ratio,
        fch_evm_db=fch_evm, data_evm_db=qam_evm,
        phase_from=phase_from,
    )
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {a.out}/symbols.npz, summary.json")


if __name__ == "__main__":
    sys.exit(main())
