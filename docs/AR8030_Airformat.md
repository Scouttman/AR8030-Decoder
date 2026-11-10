# AR8030 Air Format

The Artosyn AR8030 downlink, layer by layer, from the OFDM waveform to the bytes an application sent. All values were measured from captured signals at the default rate (16-QAM data).

| Layer | Summary |
|---|---|
| [1. Air](#1-on-the-air) | OFDM bursts on one fixed channel, ≈ 4.73 MHz wide |
| [2. Burst](#2-burst-layout) | 2 sync, 2 FCH, 24 data symbols |
| [3. Cells](#3-cells-to-coded-bits) | 16-QAM on ≈ 385 data tones, sign-scrambled, interleaved |
| [4. FEC](#4-forward-error-correction) | 8 × LDPC (2304, 1728) rate 3/4 per package |
| [5. Package](#5-package) | 1728 bytes; two per burst |
| [6. Stream](#6-user-byte-stream) | User bytes concatenated per port |
| [7. Application](#7-application-framing-example) | Sender-defined; example: chunked H.265 |

## 1. On the air

| Parameter | Value |
|---|---|
| Sample rate | 5.6 MS/s |
| FFT size | 512 |
| Cyclic prefix | 32 samples (1/16) |
| Tone spacing | 10.9375 kHz |
| Used tones | −215 … +217, DC empty |
| Occupied width | ≈ 4.73 MHz |
| Burst length | ≈ 2.81 ms |
| Burst period | ≈ 5.63 ms |

FFT size, sample rate, tone spacing and cyclic prefix are the IEEE 802.16e (WiMAX) 5 MHz profile. A few later building blocks also come from 802.16e and are noted where they appear. The format as a whole is not 802.16; a WiMAX receiver will not decode it.

One fixed channel, no frequency hopping. The tone grid has one more tone above DC than below, so the grid centre is offset from the channel centre by a few kHz.

## 2. Burst layout

```
 2 × 512     2 × BPSK            24 × (512 + 32 CP), 16-QAM
 no CP
+----+----+----+----+---------------------------+---------------------------+
|sync|sync|FCH |FCH | data symbols 0–11         | data symbols 12–23        |
|    |    |    |    | package k                 | package k+1               |
+----+----+----+----+---------------------------+---------------------------+
|<----------------------------- ≈ 2.81 ms --------------------------------->|
```

Each half of the data region carries one complete package.

### Sync header

One 512-sample BPSK symbol on odd tones only, sent twice with no cyclic prefix. Identical in every burst; carries no rate, counter or length. Not the 802.16 preamble, which is one symbol on every third tone. The 217 odd tones, lowest first, +1 → 1:

```
0x39454d4088633883bcd93b6cbf36e86bf3f9765fd6dc0bde437500
```

Frequency offset from the two copies:

```
cfo = −angle(Σ z[n]·conj(z[n+512])) / (2π·512 / fs)
```

Data symbol 0 starts directly after the second copy.

### FCH

Two BPSK symbols carrying the rate setting and a 3-bit burst counter. The two symbols are different encodings of the same bits. The code is not identified. The payload can be decoded without it. The name follows 802.16; the content does not.

### Pilots

Every 18th tone. The position (tone mod 18) cycles `0, 9, 12, 3, 15, 6` over successive symbols. The tone 9 above each pilot carries no data. Pilots are BPSK on the real or imaginary axis depending on the symbol, with a fixed pattern that repeats in every burst. That leaves 385 data tones per symbol (384 on symbols where tone −216 falls on a pilot or gap). This is not one of the 802.16 allocations (PUSC, FUSC, AMC).

## 3. Cells to coded bits

Each data tone is one 16-QAM point: sign and magnitude bits on I and Q. Twelve symbols give ≈ 18 480 cells for the 18 432 coded bits of one package.

### Cell-sign scrambling

- I and Q sign bits are XORed with the m-sequence of `1 + x² + x¹¹`, indexed by `k − s` (tone index minus symbol index). Magnitude bits are not scrambled. The polynomial is the 802.16 subcarrier randomizer `1 + x⁹ + x¹¹` reversed; the way it is applied differs.
- Each symbol also has a fixed I flip and Q flip, which depend on the demodulator's phase reference.

### Interleaver

A fixed permutation from coded bits to (symbol, tone, bit position). It does not change between bursts, sessions or devices. No complete formula is known; it is available as a measured table. Its structure:

1. **Sub-blocks.** Each 2304-bit codeword is cut into three 768-bit sub-blocks: 24 per package. All 24 use the same bit pattern.
2. **Bit streams.** Within a sub-block, bit *k* belongs to stream *k* mod 4. Each 16-QAM point takes one bit from each stream: streams 0 and 2 give the magnitude bits, 1 and 3 the sign bits; streams 0 and 1 share one rail, 2 and 3 the other. Each stream is permuted separately (192 positions). The four permutations are not affine, QPP or the 802.16 bit interleaver (all 802.16 block sizes tested).
3. **Points to tones.** All four bits of a tone come from one sub-block. Along the data tones of a symbol the owning sub-block advances by 7, 7, 7, 6 (mod 24), an average of 27/4.
4. **Per-symbol rotation.** Each half of the band (below and above DC) wraps at a point that moves with a period of 3 symbols (tones −108, −162, −54 and 55, 163, 109).
5. **Codeword offsets.** Codeword *c*+4 uses the tones of codeword *c*, 6 symbols later. Codeword *c*+2 is ≈ 3 symbols later.
6. **Second package.** Package k+1 uses the same map 12 symbols later, with I and Q swapped.

## 4. Forward error correction

A package is 1728 information bytes in 8 codewords of 216 bytes. Transmit order:

1. **Randomize:** XOR the 13 824 information bits with the PRBS `1 + x¹⁴ + x¹⁵`, one continuous stream over all 8 codewords, MSB first. Same in every burst. First 15 bits: `000000111111011`. The polynomial is the 802.16 one; 802.16 reseeds per burst, this does not.
2. **Encode:** IEEE 802.16e LDPC, n = 2304, k = 1728, rate 3/4 B, z = 96, systematic.
3. **Invert** all 576 parity bits. Not part of 802.16.
4. **Map** with sign scrambling and the interleaver.

## 5. Package

Own format, unrelated to the 802.16 MAC.

| Bytes | Field | Notes |
|---|---|---|
| 0–1 | Length | Big-endian; package size − 5. 44 with no user data. |
| 4 b7–5 | Burst counter | Mod 8. |
| 7–8, 10, 14 | Rotating fields | Period 5; with the counter, period 40. Meaning unknown. |
| 12–15 | User data header | Only when user data is present: `01 <port+1> L0 L1`. Length = `(L1<<8 \| L0) >> 4`. |
| 16–52 | Link fields | Mostly constant (start at byte 12 when there is no user header). |
| 53… | User data | Up to 1672 bytes, then zero padding. |
| 1725 | Reserved | Zero. |
| 1726–1727 | CRC-16 | CRC-16/XMODEM (poly 0x1021, init 0, no xorout) over bytes 0–1725, big-endian. |

## 6. User byte stream

Packages are in order burst *n* first half, burst *n* second half, burst *n*+1 first half, and so on. Concatenating the user data of a port, skipping packages without a user header, gives the sender's byte stream. Package boundaries are independent of the sender's writes. There is no retransmission. Capacity is 2 × 1672 bytes per burst, ≈ 4.75 Mbit/s.

## 7. Application framing (example)

The stream content is defined by the sending software. One open-source video sender uses 22-byte little-endian chunk headers:

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | Magic | `AFRM` |
| 4 | 2 | Frame sequence | |
| 6 | 2 | Chunk index | |
| 8 | 2 | Chunk count | |
| 10 | 2 | Payload length | |
| 12 | 1 | Flags | Bit 0 keyframe, bit 1 intra-refresh. |
| 13 | 1 | Codec | 1 H.265, 2 Opus, 3 control, 4 telemetry. |
| 14 | 4 | Timestamp | µs. |
| 18 | 2 | Payload CRC | CRC-16/CCITT-FALSE over the payload. |
| 20 | 2 | Reserved | |

Video payloads are Annex-B H.265.

## Not yet known

- The four per-stream permutations inside a sub-block, and the exact tone rule. Steps 1, 2, 5 and 6 of the interleaver are rules; steps 3 and 4 are partly described; the full map exists only as a measured table.
- The FCH code and most of its bits beyond rate and counter.
- Bit placement at rates other than the default 16-QAM setting (QPSK rates not mapped).
- The rotating header fields and most link fields.
- A few header bits in codewords 0 and 7 that never changed during testing and have no measured cell. Decoders treat them as erasures.
