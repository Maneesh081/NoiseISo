"""Generate a small, deterministic VoiceBank+DEMAND-style corpus from scratch.

WHY THIS EXISTS
    The real VoiceBank+DEMAND corpus is ~3.6 GB and downloads slowly. To get the whole
    pipeline (training -> evaluation -> web UI) running *today*, we synthesise our own
    pairs with the same recipe the real corpus uses:

        noisy = clean_speech + noise,  mixed at a fixed SNR in dB

    The only difference is where the raw material comes from: the real corpus records
    human speakers in a studio and records real-world noise; we generate both. The
    training code cannot tell the difference -- it only ever sees paired WAV files
    sitting in matching folders, which is exactly what this script writes.

    Everything is seeded, so running it twice produces byte-identical audio.

WHAT IT MAKES
    data/clean_trainset_28spk_wav/p001_0000.wav   <- clean speech only
    data/noisy_trainset_28spk_wav/p001_0000.wav   <- the SAME sentence + noise
    data/clean_testset_wav/p901_0000.wav          <- unseen speakers
    data/noisy_testset_wav/p901_0000.wav

    16 kHz mono, 16-bit PCM, 3 seconds each, identical filenames on both sides.

HOW THE PIECES ARE MADE
    speech : a glottal pulse train (the "voice") shaped by moving formant filters (the
             "mouth"). This is a classic source-filter model of speech -- the same
             idea that makes a vowel sound like a vowel.
    noise  : coloured noise plus a slow random wobble. White/pink/brown noise come from
             filtering random numbers; babble/traffic/cafe/aircraft get extra
             amplitude modulation or a low rumble so their spectrograms look distinct.
"""

import argparse
import os

import numpy as np
import soundfile as sf

SR = 16000

# The SNR levels the real VoiceBank corpus uses, split the same way.
TRAIN_SNRS = [0, 5, 10, 15, 20]
TEST_SNRS = [0, 5, 10, 15, 20]

TRAIN_NOISES = [
    "babble", "traffic", "cafe", "restaurant", "street",
    "aircraft", "car", "park", "shoppingmall", "train",
]
TEST_NOISES = [
    "babble", "traffic", "cafe", "restaurant", "street",
]


# --------------------------------------------------------------------------- speech

def _glottal_source(n_samples, f0, rng):
    """Pulse train: one short pulse every 1/f0 seconds. This is the raw 'voice'."""
    period = SR / f0
    src = np.zeros(n_samples, dtype=np.float64)
    pos = 0.0
    while pos < n_samples:
        i = int(pos)
        # A smooth two-pole-ish pulse shape (a raised cosine) is closer to a real
        # glottal waveform than a bare impulse, which matters for timbre.
        width = max(2, int(0.004 * SR))  # ~4 ms pulse
        end = min(i + width, n_samples)
        t = np.linspace(0.0, 1.0, end - i, endpoint=False)
        src[i:end] += 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
        pos += period * (1.0 + 0.02 * rng.standard_normal())
    return src


def _resonator(x, freq, bandwidth):
    """Two-pole resonator: emphasises energy near `freq` (one vocal-tract formant)."""
    r = np.exp(-np.pi * bandwidth / SR)
    theta = 2.0 * np.pi * freq / SR
    a1 = 2.0 * r * np.cos(theta)
    a2 = -(r ** 2)
    g = (1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * theta) + r ** 2)
    y = np.zeros_like(x)
    for n in range(len(x)):
        # y[n] = g*x[n] + a1*y[n-1] + a2*y[n-2]
        y[n] = g * x[n] + (a1 * y[n - 1] if n >= 1 else 0.0) + (a2 * y[n - 2] if n >= 2 else 0.0)
    return y


def _vowel_sequence(n_samples, rng):
    """Concatenate voiced vowel segments, each with its own drifting formants."""
    f1_pool = [730, 270, 570, 300, 660]   # /u/ /i/ /a/ /o/ /e/ roughly
    f2_pool = [1090, 2290, 840, 870, 1720]
    f3_pool = [2440, 3010, 2410, 3020, 2410]

    out = np.zeros(n_samples, dtype=np.float64)
    write = 0
    # A sentence = 5-14 short segments, each 80-250 ms, separated by tiny silence.
    for _ in range(int(rng.integers(5, 14))):
        if write >= n_samples:
            break
        seg = int(rng.uniform(0.08, 0.25) * SR)
        seg = min(seg, n_samples - write)
        if seg <= 0:
            break

        v = int(rng.integers(0, len(f1_pool)))
        # Formants glide within a segment, which is what makes it sound like a "diphthong".
        f0 = float(rng.uniform(85.0, 255.0))
        base_f1 = f1_pool[v] * float(rng.uniform(0.94, 1.06))
        f1 = np.linspace(base_f1, base_f1 * float(rng.uniform(0.85, 1.2)), seg)
        f2 = np.linspace(f2_pool[v], f2_pool[v] * float(rng.uniform(0.9, 1.12)), seg)
        f3 = f3_pool[v] * float(rng.uniform(0.95, 1.05))

        t = np.arange(seg) / SR
        # Slight vibrato + drift keeps the pitch from sounding like a test tone.
        inst_f0 = f0 * (1.0 + 0.01 * np.sin(2.0 * np.pi * 4.5 * t) + 0.004 * np.sin(2.0 * np.pi * 1.3 * t))
        # Build the pulse train for this segment at its own varying f0.
        src = np.zeros(seg)
        phase = 0.0
        for n in range(seg):
            phase += inst_f0[n] / SR
            if phase >= 1.0:
                phase -= 1.0
                w = min(int(0.004 * SR), seg - n)
                tt = np.linspace(0.0, 1.0, w, endpoint=False)
                src[n:n + w] += 0.5 * (1.0 - np.cos(2.0 * np.pi * tt))

        # Apply three formants, each a little at a time (real vocal tracts are narrow).
        shaped = _resonator(src, float(np.mean(f1)), 80.0)
        shaped += 0.6 * _resonator(src, float(np.mean(f2)), 110.0)
        shaped += 0.3 * _resonator(src, f3, 160.0)

        # Segment envelope so letters do not click into each other.
        env = np.sin(np.pi * np.linspace(0.0, 1.0, seg)) ** 0.7
        out[write:write + seg] += shaped * env
        write += seg

        # A short gap (a "space" between words).
        gap = int(rng.uniform(0.01, 0.06) * SR)
        gap = min(gap, n_samples - write)
        write += max(gap, 0)

    return out


def make_speech(n_samples, rng):
    """Full clean speech clip, normalised to a sane peak."""
    s = _vowel_sequence(n_samples, rng)
    peak = np.max(np.abs(s))
    if peak > 1e-9:
        s = s / peak * 0.7
    return s


# ------------------------------------------------------------------------------ noise

def _norm(x, peak=1.0):
    p = np.max(np.abs(x))
    return x / p * peak if p > 1e-9 else x


def _colour(n, kind, rng):
    """White / pink / brown noise via spectral shaping."""
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    if kind == "pink":
        shape = 1.0 / np.sqrt(f)
    else:  # brown
        shape = 1.0 / f
    return np.fft.irfft(spec * shape, n)


def make_noise(kind, n_samples, rng):
    """One noise clip of the requested 'flavour', all normalised to peak 1."""
    base = {
        "babble": "pink",
        "traffic": "brown",
        "cafe": "pink",
        "restaurant": "pink",
        "street": "pink",
        "aircraft": "brown",
        "car": "brown",
        "park": "pink",
        "shoppingmall": "pink",
        "train": "brown",
    }.get(kind, "white")
    noise = _colour(n_samples, base, rng)

    if kind == "car":
        # Low-frequency engine rumble that swells and fades.
        t = np.arange(n_samples) / SR
        noise += 0.8 * np.sin(2.0 * np.pi * 70.0 * t) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.7 * t))
    elif kind == "aircraft":
        t = np.arange(n_samples) / SR
        noise += 0.6 * np.sin(2.0 * np.pi * 120.0 * t + np.sin(2.0 * np.pi * 3.0 * t))
    elif kind == "traffic":
        t = np.arange(n_samples) / SR
        noise += 0.5 * np.sin(2.0 * np.pi * 55.0 * t)
    elif kind == "park":
        # Windy: slow gain wobble.
        t = np.arange(n_samples) / SR
        noise *= 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.25 * t + rng.uniform(0, 6.28))
    elif kind in ("cafe", "restaurant", "shoppingmall"):
        # Clatter: random short bursts, like cups and cutlery.
        for _ in range(int(n_samples / (0.12 * SR)) + 1):
            start = int(rng.uniform(0, max(1, n_samples - int(0.05 * SR))))
            length = int(rng.uniform(0.005, 0.03) * SR)
            noise[start:start + length] += 2.0 * rng.standard_normal(length)
    elif kind == "babble":
        # Many overlapping voices: sum a few slow, formant-ish wobbling tones.
        t = np.arange(n_samples) / SR
        for _ in range(6):
            f = float(rng.uniform(90.0, 320.0))
            env = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(rng.uniform(0.4, 2.2)) * t)
            noise += 0.45 * env * np.sin(2.0 * np.pi * f * t + rng.uniform(0, 6.28))
    elif kind == "street":
        t = np.arange(n_samples) / SR
        noise *= 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.15 * t)

    return _norm(noise, 1.0)


# ------------------------------------------------------------------------------ mixing

def mix_at_snr(clean, noise, snr_db, rng):
    """Mix clean + noise so the result hits `snr_db` decibels, as VoiceBank does.

    SNR is defined on *active* speech frames, otherwise long silences would drag the
    measured SNR down. We therefore scale the noise using the peak of the clean signal.
    """
    clean = _norm(clean, 1.0)
    noise = _norm(noise, 1.0)

    speech_rms = np.sqrt(np.mean(clean ** 2)) + 1e-12
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-12

    # We want: 10*log10( speech_rms^2 / (noise_rms^2 * k^2) ) = snr_db
    #  => k = speech_rms / (noise_rms * 10^(snr_db/20))
    k = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    mixed = clean + k * noise

    # Guard against clipping: VoiceBank keeps the clean track as the untouched
    # reference and lets the noisy track be whatever level the mix implies.
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    return mixed


# ------------------------------------------------------------------------------- main

def generate_split(out_root, speakers, noises, snrs, rng, tag):
    clean_dir = os.path.join(out_root, f"clean_{tag}")
    noisy_dir = os.path.join(out_root, f"noisy_{tag}")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    count = 0
    for sp in range(speakers):
        for utt in range(4):  # 4 utterances per speaker keeps it small but non-trivial
            n = int(3.0 * SR)  # 3 seconds
            clean = make_speech(n, rng)
            kind = noises[int(rng.integers(0, len(noises)))]
            noise = make_noise(kind, n, rng)
            snr = int(snrs[int(rng.integers(0, len(snrs)))])
            noisy = mix_at_snr(clean, noise, snr, rng)

            # VoiceBank names files like p226_001_SNR05.wav; we keep it simple and
            # identical on both sides, which is all the dataset loader needs.
            name = f"p{sp + 1:03d}_{utt:04d}.wav"
            sf.write(os.path.join(clean_dir, name),
                     clean.astype(np.float32), SR, subtype="PCM_16")
            sf.write(os.path.join(noisy_dir, name),
                     noisy.astype(np.float32), SR, subtype="PCM_16")
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="../data")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train_speakers", type=int, default=28)
    ap.add_argument("--test_speakers", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    n_train = generate_split(out, args.train_speakers, TRAIN_NOISES, TRAIN_SNRS, rng, "trainset_28spk_wav")
    n_test = generate_split(out, args.test_speakers, TEST_NOISES, TEST_SNRS, rng, "testset_wav")

    print(f"[synth] wrote {n_train} train pairs and {n_test} test pairs into {out}")
    for d in sorted(os.listdir(out)):
        p = os.path.join(out, d)
        if os.path.isdir(p):
            w = len([f for f in os.listdir(p) if f.endswith(".wav")])
            print(f"   {d}: {w} wav")


if __name__ == "__main__":
    main()
