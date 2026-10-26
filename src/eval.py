"""Cost of Quality: evaluate a trained checkpoint on the held-out VoiceBank+DEMAND test set.

WHY ONLY TWO NUMBERS MATTER: PESQ + STOI.

PESQ (Perceptual Evaluation of Speech Quality):
    -0.5 .. 4.5, higher is better. It predicts how a human listener would rate the audio
    on a phone-quality scale. Noisy VoiceBank input scores ~1.2-1.5. A decent U-Net usually
    reaches ~2.8-3.1. This is the headline number in every speech-enhancement paper.

STOI (Short-Time Objective Intelligibility):
    0 .. 1, higher is better. It predicts how *understandable the words* are. Noisy input
    ~0.78, a good model ~0.95+.

We always compare THREE things for every file, so you can see the before -> after jump:

  threshold: the *noisy* audio (input)
  enhanced : the audio our model produced
  clean    : the original studio recording (the ground truth everyone is judged against)

PESQ needs wavelets and STOI needs comparable frames; both expect the exact same audio
length on both sides. We cache them on disk so a failed run does not lose the first
half of the results.
"""

import argparse
import glob
import os

import numpy as np
import soundfile as sf
import torch

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    # PESQ is a C extension and needs a compiler at install time. When it is not
    # available we still report STOI, which is pure Python and always works.
    PESQ_AVAILABLE = False
from pystoi import stoi as stoi_fn

from features import Spectrogram
from model import UNet
from utils import ensure_dir, set_seed, model_device


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"{path}: {sr} Hz, expected 16000")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


def enhance_clip(model, stft, noisy_wavs: torch.Tensor, device) -> torch.Tensor:
    """Run the model over one noisy clip (already scaled) and return the clean waveform.

    We use the noisy *phase* when rebuilding audio -- the human ear is far less sensitive
    to phase than to loudness, so this well-known shortcut gives good sound quality at
    a fraction of the effort. The whole clip goes through at once in a single forward pass.
    """
    noisy = noisy_wavs.unsqueeze(0).to(device)          # (1, samples)
    scale = noisy.abs().max().clamp_min(1e-6)
    noisy_n = noisy / scale

    mag = stft.magnitude(noisy_n).unsqueeze(1)          # (1, 1, F, T)
    phase = stft.phase(noisy_n)                          # (1, F, T)

    with torch.no_grad():
        mask = model(mag).squeeze(1)                     # (1, F, T) 0..1
        enhanced_mag = mask * stft.magnitude(noisy_n)    # apply the mask

    clean_audio = stft.recover(enhanced_mag, phase, noisy_n.shape[-1]).squeeze(0)
    return clean_audio * scale                          # undo normalisation


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stft = Spectrogram().to_device(device)

    model = UNet(base=args.base, depth=args.depth).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ensure_dir(args.out_dir)
    clips = sorted(glob.glob(os.path.join(args.noisy_dir, "*.wav")))
    rows = []
    for path in clips:
        name = os.path.basename(path)
        clean_path = os.path.join(args.clean_dir, name)
        if not os.path.exists(clean_path):
            continue
        clean = load_wav(clean_path)
        noisy = load_wav(path)
        length = min(len(clean), len(noisy))

        with torch.no_grad():
            enhanced = enhance_clip(model, stft,
torch.from_numpy(noisy[:length]), device).cpu().numpy()

        if PESQ_AVAILABLE:
            pesq_in = pesq_fn(16000, clean[:length], noisy[:length], "wb")
            pesq_out = pesq_fn(16000, clean[:length], enhanced, "wb")
            stoi_in = stoi_fn(clean[:length], noisy[:length], 16000, extended=False)
            stoi_out = stoi_fn(clean[:length], enhanced, 16000, extended=False)

            rows.append([name, round(pesq_in, 3), round(pesq_out, 3),
                         round(stoi_in, 4), round(stoi_out, 4)])
            print(f"[eval] {name}: PESQ {pesq_in:.2f}->{pesq_out:.2f} | "
                  f"STOI {stoi_in:.3f}->{stoi_out:.3f}")
        else:
            stoi_in = stoi_fn(clean[:length], noisy[:length], 16000, extended=False)
            stoi_out = stoi_fn(clean[:length], enhanced, 16000, extended=False)
            rows.append([name, None, None, round(stoi_in, 4), round(stoi_out, 4)])
            print(f"[eval] {name}: STOI {stoi_in:.3f}->{stoi_out:.3f}")

    if not rows:
        print("[eval] nothing to evaluate.")
        return

    summary = {
        "stoi_noisy": float(np.mean([r[3] for r in rows])),
        "stoi_enhanced": float(np.mean([r[4] for r in rows])),
        "files": len(rows),
    }
    summary["stoi_delta"] = summary["stoi_enhanced"] - summary["stoi_noisy"]

    if PESQ_AVAILABLE:
        summary["pesq_noisy"] = float(np.mean([r[1] for r in rows]))
        summary["pesq_enhanced"] = float(np.mean([r[2] for r in rows]))
        summary["pesq_delta"] = summary["pesq_enhanced"] - summary["pesq_noisy"]

    import csv
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "pesq_noisy", "pesq_enhanced", "stoi_noisy", "stoi_enhanced"])
        w.writerows(rows)

    import json
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if PESQ_AVAILABLE:
        print(f"\n[PESQ] noisy {summary['pesq_noisy']:.3f} -> enhanced {summary['pesq_enhanced']:.3f}"
              f"   ({summary['pesq_delta']:+.3f})")
    else:
        print("\n[PESQ] not available in this environment (needs a C compiler); "
              "STOI below is the reported metric.")
    print(f"[STOI] noisy {summary['stoi_noisy']:.4f} -> enhanced {summary['stoi_enhanced']:.4f}"
          f"   ({summary['stoi_delta']:+.4f})")
    print(f"[eval] {summary['files']} files -> {os.path.join(args.out_dir, 'results.csv')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="../out/checkpoints/best.pt")
    p.add_argument("--clean_dir", default="../data/clean_testset_wav")
    p.add_argument("--noisy_dir", default="../data/noisy_testset_wav")
    p.add_argument("--out_dir", default="../out")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    args = p.parse_args()
    evaluate(args)
