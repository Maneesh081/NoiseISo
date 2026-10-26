"""Turn the real VoiceBank+DEMAND corpus into the 16 kHz layout the code expects.

The official VoiceBank+DEMAND wav files are 48 kHz. This project models speech at
16 kHz (see DL.md), and dataset.py / eval.py require 16 kHz mono input, so the
downloaded corpus must be resampled once before training.

It takes a staging folder (--src) that holds EITHER:

  * the four individual zips:
        clean_trainset_28spk_wav.zip
        noisy_trainset_28spk_wav.zip
        clean_testset_wav.zip
        noisy_testset_wav.zip
  * the combined DS_10283_2791.zip ("download all files"), or
  * the folders already extracted (clean_trainset_28spk_wav/, ... anywhere below).

Zips are extracted automatically (the combined one is extracted twice as needed),
then every wav in the four known sets is resampled to 16 kHz mono and written into
the project layout:

    data/clean_trainset_28spk_wav/p226_001.wav
    data/noisy_trainset_28spk_wav/p226_001.wav
    data/clean_testset_wav/p232_001.wav
    data/noisy_testset_wav/p232_001.wav

Validation split (optional)
    --val_speakers p286,p287 moves those speakers out of the training folders and
    into data/clean_val_wav and data/noisy_val_wav, so the official 824 test clips
    (p232, p257) stay purely held out for evaluation.

Filenames are preserved exactly, so the clean and noisy members of a pair keep
matching names (which is how dataset.py pairs them).

Usage
    python prepare_real_data.py --src ../data/_raw --dst ../data --val_speakers p286,p287

Safe to re-run: already-16 kHz inputs are copied through unchanged.
"""

import argparse
import glob
import os
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

TARGET_SR = 16000

# set name -> (is training set, validation set name)
SETS = {
    "clean_trainset_28spk_wav": "clean_val_wav",
    "noisy_trainset_28spk_wav": "noisy_val_wav",
    "clean_testset_wav": None,
    "noisy_testset_wav": None,
}


def extract_archives(src):
    """Extract every zip/tar under src (including archives inside archives)."""
    processed = set()
    for _ in range(6):
        archives = []
        for root, _dirs, files in os.walk(src):
            for f in files:
                if f.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                    archives.append(os.path.join(root, f))
        new = [p for p in archives if p not in processed]
        if not new:
            break
        for path in new:
            out = os.path.join(os.path.dirname(path),
                               os.path.splitext(os.path.basename(path))[0] + "_x")
            os.makedirs(out, exist_ok=True)
            print(f"[extract] {path} -> {out}")
            try:
                if path.lower().endswith(".zip"):
                    with zipfile.ZipFile(path) as zf:
                        zf.extractall(out)
                else:
                    with tarfile.open(path) as tf:
                        tf.extractall(out)
            except (zipfile.BadZipFile, tarfile.TarError) as e:
                print(f"[extract] FAILED {path}: {e}")
            processed.add(path)


def find_set_dir(src, set_name):
    """Locate the folder with this basename anywhere under the staging root."""
    for root, dirs, _ in os.walk(src):
        if set_name in dirs:
            return os.path.join(root, set_name)
    return None


def convert_one(task):
    """Resample one file. Runs in a worker process."""
    path, out_dir = task
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:                       # safety: downmix to mono
        wav = wav.mean(axis=1)
    was_resampled = sr != TARGET_SR
    if was_resampled:
        import librosa
        wav = librosa.resample(wav.astype(np.float32),
                               orig_sr=sr, target_sr=TARGET_SR)
    out_path = os.path.join(out_dir, os.path.basename(path))
    sf.write(out_path, np.ascontiguousarray(wav, dtype=np.float32),
             TARGET_SR, subtype="PCM_16")
    return was_resampled


def process_set(in_dir, dst, set_name, val_speakers, workers):
    val_name = SETS[set_name]
    train_out = os.path.join(dst, set_name)
    val_out = os.path.join(dst, val_name) if val_name else None
    os.makedirs(train_out, exist_ok=True)
    if val_out:
        os.makedirs(val_out, exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    tasks = []
    val_count = 0
    for path in wavs:
        speaker = os.path.basename(path).split("_")[0]
        to_val = bool(val_out and val_speakers and speaker in val_speakers)
        tasks.append((path, val_out if to_val else train_out))
        val_count += int(to_val)

    resampled = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, was_resampled in enumerate(ex.map(convert_one, tasks), 1):
            resampled += int(was_resampled)
            if i % 1000 == 0 or i == len(tasks):
                print(f"  {set_name}: {i}/{len(tasks)}", flush=True)

    msg = f"[done] {set_name}: {len(wavs)} files ({resampled} resampled)"
    if val_out:
        msg += f", {val_count} -> {val_name}"
    print(msg, flush=True)
    return len(wavs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="../data/_raw",
                   help="staging folder with the zips or extracted sets")
    p.add_argument("--dst", default="../data",
                   help="project data folder to write the 16 kHz sets into")
    p.add_argument("--val_speakers", default="",
                   help="comma-separated train speakers to hold out for validation")
    p.add_argument("--no-extract", action="store_true",
                   help="skip zip extraction (sets are already extracted)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel file-conversion workers (default 4)")
    args = p.parse_args()

    val_speakers = {s.strip() for s in args.val_speakers.split(",") if s.strip()}
    if val_speakers:
        print(f"[prepare] validation speakers: {', '.join(sorted(val_speakers))}")

    if not args.no_extract:
        extract_archives(args.src)

    total = 0
    for name in SETS:
        in_dir = find_set_dir(args.src, name)
        if in_dir is None:
            print(f"[skip] {name}: not found under {args.src}")
            continue
        total += process_set(in_dir, args.dst, name, val_speakers, args.workers)
    print(f"[prepare] wrote {total} files to {args.dst}")


if __name__ == "__main__":
    main()
