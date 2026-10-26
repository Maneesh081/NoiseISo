"""Inference and rendering engine shared by the FastAPI dashboard.

Everything the web UI needs lives here so the HTTP layer in app.py stays thin:
loading the checkpoint, cleaning audio (overlap-add chunking), rendering
spectrograms / waveforms / charts as base64 PNGs, and gathering the dataset and
score metadata shown on the page.

Only numpy / torch / soundfile / matplotlib are used; no Gradio.
"""

import base64
import csv
import glob
import io
import json
import math
import os
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from features import Spectrogram
from model import UNet, count_parameters

SAMPLE_RATE = 16000

# Dark palette that matches the dashboard CSS.
DARK_BG = "#0e131a"
PANEL_BG = "#131a23"
GRID = "#2a3644"
TEXT = "#c9d4df"
MUTED = "#8b98a5"
NOISY = "#8b98a5"
CLEAN = "#4cc9f0"
ACCENT = "#7c5cff"
WARN = "#ffb86b"

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a")


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def load_checkpoint(path, model):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    return model


def to_model_rate(wav, sr):
    """Convert any sample rate / channel count to 16 kHz mono float32."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype=np.float32)


class Streamer:
    """Clean audio in overlapping chunks so long clips still fit in GPU memory.

    A Hann ramp fades each chunk in/out and the overlapping regions are
    averaged, which hides the seams (same idea the STFT uses to rebuild audio).
    """

    def __init__(self, model, stft, device, chunk_seconds=4.0, overlap=0.5):
        self.model = model
        self.stft = stft
        self.device = device
        self.chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        self.hop = int(self.chunk_samples * (1.0 - overlap))

    def _chunk_mags(self, wav):
        x = torch.from_numpy(wav)
        scale = x.abs().max().clamp_min(1e-6)
        x = (x / scale).unsqueeze(0).to(self.device)
        mag = self.stft.magnitude(x).squeeze(0)
        phase = self.stft.phase(x).squeeze(0)
        return mag, phase, scale

    def _enhance_chunk(self, mag):
        x = mag.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask = self.model(x).squeeze(0)
            enhanced = (mask * mag.unsqueeze(0)).squeeze(0)
        return enhanced

    def clean(self, wav):
        if len(wav) == 0:
            return wav.astype(np.float32)
        n = math.ceil((len(wav) - self.chunk_samples) / self.hop) + 1
        n = max(n, 1)
        out = np.zeros(len(wav), dtype=np.float64)
        weight = np.zeros(len(wav), dtype=np.float64)
        for i in range(n):
            start = i * self.hop
            end = min(start + self.chunk_samples, len(wav))
            seg = wav[start:end]
            length = len(seg)
            mag, phase, scale = self._chunk_mags(seg)
            enhanced_mag = self._enhance_chunk(mag)
            seg_out = self.stft.recover(
                enhanced_mag.unsqueeze(0), phase.unsqueeze(0), length
            ).squeeze(0).cpu().numpy() * scale.item()
            ramp = np.hanning(length)
            out[start:end] += seg_out * ramp
            weight[start:end] += ramp
        weight = np.clip(weight, 1e-6, None)
        return (out / weight).astype(np.float32)


# --------------------------------------------------------------------------- #
# encoding helpers
# --------------------------------------------------------------------------- #
def _png_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _wav_bytes(wav, sr=SAMPLE_RATE):
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(wav, dtype=np.float32), sr,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _wav_data_uri(wav, sr=SAMPLE_RATE):
    return "data:audio/wav;base64," + base64.b64encode(_wav_bytes(wav, sr)).decode("ascii")


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _new_fig(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax


def _save_fig(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def spectrogram_png(wav, stft, max_frames=900):
    """Render a dB spectrogram on the dark theme."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    scale = x.abs().max().clamp_min(1e-6)
    mag = stft.magnitude(x.unsqueeze(0).to("cpu")).squeeze(0).numpy()
    if mag.shape[1] > max_frames:  # keep wide files responsive
        step = int(math.ceil(mag.shape[1] / max_frames))
        mag = mag[:, ::step]
    mag_db = 20.0 * np.log10(mag / max(scale.item(), 1e-9) + 1e-9)

    duration = len(wav) / SAMPLE_RATE
    fig, ax = _new_fig(6.2, 2.3)
    ax.imshow(mag_db, aspect="auto", origin="lower", cmap="magma",
              extent=[0, duration, 0, SAMPLE_RATE / 2000],
              vmin=mag_db.max() - 70, vmax=mag_db.max())
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.grid(False)
    return _save_fig(fig)


def waveform_png(noisy, cleaned):
    """Two stacked waveforms (noisy on top, cleaned below) sharing a time axis."""
    step = max(1, len(noisy) // 120000)  # keep long clips cheap to render
    noisy_p = noisy[::step]
    cleaned_p = cleaned[::step]
    t = np.arange(len(noisy_p)) * step / SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 2.2), dpi=120, sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    pairs = ((noisy_p, NOISY, "noisy"), (cleaned_p, CLEAN, "cleaned"))
    for ax, (wav, colour, label) in zip(axes, pairs):
        ax.set_facecolor(PANEL_BG)
        ax.plot(t, wav, color=colour, linewidth=0.5)
        ax.set_ylabel(label, color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    axes[-1].xaxis.label.set_color(MUTED)
    return _save_fig(fig)


def _score_bars(scores):
    fig, ax = _new_fig(4.0, 2.6)
    labels = ["Noisy", "Enhanced"]
    values = [scores.get("stoi_noisy", 0.0), scores.get("stoi_enhanced", 0.0)]
    bars = ax.bar(labels, values, color=[NOISY, CLEAN], width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("STOI")
    ax.set_title("Intelligibility (STOI)", fontsize=10)
    ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
    return _save_fig(fig)


def _delta_hist(rows):
    deltas = [r["enhanced"] - r["noisy"] for r in rows]
    if not deltas:
        return None
    fig, ax = _new_fig(4.0, 2.6)
    ax.hist(deltas, bins=30, color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("STOI change per file")
    ax.set_ylabel("files")
    ax.set_title("Per-file improvement", fontsize=10)
    ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
    return _save_fig(fig)


def _scatter(rows):
    if not rows:
        return None
    noisy = [r["noisy"] for r in rows]
    enhanced = [r["enhanced"] for r in rows]
    fig, ax = _new_fig(4.0, 2.6)
    ax.scatter(noisy, enhanced, s=10, color=CLEAN, alpha=0.65, edgecolors="none")
    lo = min(min(noisy), min(enhanced))
    hi = max(max(noisy), max(enhanced))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("noisy STOI")
    ax.set_ylabel("enhanced STOI")
    ax.set_title("Per-file: before vs after", fontsize=10)
    ax.grid(color=GRID, alpha=0.3, linewidth=0.6)
    return _save_fig(fig)


def _loss_curve(out_dir):
    events = sorted(glob.glob(os.path.join(out_dir, "tb", "events*")))
    if not events:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(events[-1])
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if "loss/train" not in tags:
            return None
        fig, ax = _new_fig(4.6, 2.6)
        series = (("loss/train", ACCENT, "train"), ("loss/val", CLEAN, "validation"))
        for tag, colour, name in series:
            if tag in tags:
                ev = acc.Scalars(tag)
                ax.plot([e.step for e in ev], [e.value for e in ev],
                        label=name, color=colour, linewidth=1.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("L1 loss")
        ax.set_title("Training progress", fontsize=10)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=8)
        ax.grid(color=GRID, alpha=0.3, linewidth=0.6)
        return _save_fig(fig)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class Engine:
    """Holds the loaded model and answers every question the UI asks."""

    def __init__(self, checkpoint, data_dir, out_dir, test_dir=None,
                 chunk_seconds=4.0):
        self.checkpoint = os.path.abspath(checkpoint)
        self.data_dir = os.path.abspath(data_dir)
        self.out_dir = os.path.abspath(out_dir)
        self.test_dir = os.path.abspath(test_dir) if test_dir else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stft = Spectrogram().to_device(self.device)
        self.model = UNet(base=16, depth=4).to(self.device)
        self.model = load_checkpoint(self.checkpoint, self.model)
        self.model.eval()
        self.streamer = Streamer(self.model, self.stft, self.device,
                                 chunk_seconds=chunk_seconds, overlap=0.5)
        self.lock = threading.Lock()
        self._spec_cache = {}
        self._counts_cache = None

        self.audio_sets = {
            "clean_trainset_28spk_wav": os.path.join(self.data_dir, "clean_trainset_28spk_wav"),
            "noisy_trainset_28spk_wav": os.path.join(self.data_dir, "noisy_trainset_28spk_wav"),
            "clean_val_wav": os.path.join(self.data_dir, "clean_val_wav"),
            "noisy_val_wav": os.path.join(self.data_dir, "noisy_val_wav"),
            "clean_testset_wav": os.path.join(self.data_dir, "clean_testset_wav"),
            "noisy_testset_wav": os.path.join(self.data_dir, "noisy_testset_wav"),
        }
        if self.test_dir:
            self.audio_sets["test_noisy"] = self.test_dir

    # ---- model info ------------------------------------------------------ #
    def model_info(self):
        params = count_parameters(self.model)
        try:
            size_mb = os.path.getsize(self.checkpoint) / (1024 ** 2)
        except OSError:
            size_mb = 0.0
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return {
            "params": params,
            "params_m": params / 1e6,
            "checkpoint_mb": size_mb,
            "device": str(self.device),
            "gpu": gpu,
            "sr": SAMPLE_RATE,
            "freq_bins": self.stft.freq_bins,
            "n_fft": self.stft.n_fft,
            "hop": self.stft.hop,
            "chunk_seconds": self.streamer.chunk_samples / SAMPLE_RATE,
        }

    # ---- inference ------------------------------------------------------- #
    def enhance_array(self, wav, sr):
        original_sr = sr
        wav16 = to_model_rate(wav, sr)
        with self.lock:
            cleaned = self.streamer.clean(wav16)
        peak_in = float(np.abs(wav16).max()) if len(wav16) else 0.0
        peak_out = float(np.abs(cleaned).max()) if len(cleaned) else 0.0
        rms_in = float(np.sqrt(np.mean(wav16 ** 2))) if len(wav16) else 0.0
        rms_out = float(np.sqrt(np.mean(cleaned ** 2))) if len(cleaned) else 0.0
        return {
            "original_sr": original_sr,
            "model_sr": SAMPLE_RATE,
            "resampled": original_sr != SAMPLE_RATE,
            "duration": len(cleaned) / SAMPLE_RATE,
            "samples": int(len(cleaned)),
            "noisy": wav16,
            "cleaned": cleaned,
            "stats": {
                "peak_in": peak_in,
                "peak_out": peak_out,
                "rms_in": rms_in,
                "rms_out": rms_out,
            },
        }

    def enhance_bytes(self, data):
        try:
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as exc:
            raise ValueError(f"could not decode audio: {exc}")
        if len(wav) == 0:
            raise ValueError("audio clip is empty")
        res = self.enhance_array(wav, sr)
        noisy, cleaned = res["noisy"], res["cleaned"]
        return {
            "original_sr": res["original_sr"],
            "model_sr": res["model_sr"],
            "resampled": res["resampled"],
            "duration": res["duration"],
            "samples": res["samples"],
            "stats": res["stats"],
            "noisy_audio": _wav_data_uri(noisy),
            "clean_audio": _wav_data_uri(cleaned),
            "noisy_spec": _png_data_uri(spectrogram_png(noisy, self.stft)),
            "clean_spec": _png_data_uri(spectrogram_png(cleaned, self.stft)),
            "waveform": _png_data_uri(waveform_png(noisy, cleaned)),
        }

    # ---- dataset --------------------------------------------------------- #
    def _counts(self):
        if self._counts_cache is not None:
            return self._counts_cache
        counts = {}
        for key, path in self.audio_sets.items():
            if os.path.isdir(path):
                counts[key] = sum(1 for f in os.listdir(path)
                                  if f.lower().endswith(".wav"))
            else:
                counts[key] = 0
        self._counts_cache = counts
        return counts

    def dataset(self):
        counts = self._counts()
        train_dir = self.audio_sets["clean_trainset_28spk_wav"]
        speakers = set()
        if os.path.isdir(train_dir):
            for f in os.listdir(train_dir):
                if f.lower().endswith(".wav"):
                    speakers.add(f.split("_")[0])

        samples = []
        noisy_dir = self.audio_sets["noisy_testset_wav"]
        clean_dir = self.audio_sets["clean_testset_wav"]
        if os.path.isdir(noisy_dir) and os.path.isdir(clean_dir):
            names = sorted(f for f in os.listdir(noisy_dir)
                           if f.lower().endswith(".wav"))
            if names:
                step = max(1, len(names) // 6)
                for name in names[::step][:6]:
                    if os.path.exists(os.path.join(clean_dir, name)):
                        samples.append({
                            "name": name,
                            "noisy_audio": "/api/audio?set=noisy_testset_wav&name=" + name,
                            "clean_audio": "/api/audio?set=clean_testset_wav&name=" + name,
                            "noisy_spec": "/api/spec?set=noisy_testset_wav&name=" + name,
                            "clean_spec": "/api/spec?set=clean_testset_wav&name=" + name,
                        })

        return {
            "counts": counts,
            "train_pairs": counts.get("clean_trainset_28spk_wav", 0),
            "val_pairs": counts.get("clean_val_wav", 0),
            "test_pairs": counts.get("clean_testset_wav", 0),
            "speakers": sorted(speakers),
            "num_speakers": len(speakers),
            "snr_db": [0, 5, 10, 15],
            "noise_types": ["acoustic kitchen", "babble", "cafe", "car",
                            "computer keyboard", "field", "office",
                            "public square", "station", "traffic",
                            "vacuum cleaner", "washing machine"],
            "samples": samples,
        }

    def demo_files(self):
        if not self.test_dir or not os.path.isdir(self.test_dir):
            return []
        names = sorted(f for f in os.listdir(self.test_dir)
                       if f.lower().endswith(AUDIO_EXTS))
        return [{"name": n,
                 "audio": "/api/audio?set=test_noisy&name=" + n}
                for n in names]

    def resolve_audio(self, set_name, name):
        if set_name not in self.audio_sets:
            raise KeyError("unknown audio set")
        if "/" in name or "\\" in name or ".." in name:
            raise KeyError("invalid file name")
        base = self.audio_sets[set_name]
        path = os.path.join(base, name)
        if not os.path.isfile(path) or not name.lower().endswith(AUDIO_EXTS):
            raise KeyError("file not found")
        return path

    def spectrogram_for(self, set_name, name):
        key = (set_name, name)
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        path = self.resolve_audio(set_name, name)
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        wav = to_model_rate(wav, sr)
        png = spectrogram_png(wav, self.stft)
        self._spec_cache[key] = png
        return png

    # ---- overview / scores ---------------------------------------------- #
    def _read_rows(self):
        path = os.path.join(self.out_dir, "results.csv")
        rows = []
        if not os.path.exists(path):
            return rows
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append({
                            "file": r["file"],
                            "noisy": float(r["stoi_noisy"]),
                            "enhanced": float(r["stoi_enhanced"]),
                        })
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass
        return rows

    def overview(self):
        results = _read_json(os.path.join(self.out_dir, "results.json"))
        config = _read_json(os.path.join(self.out_dir, "config.json"))
        rows = self._read_rows()

        scores = None
        if results:
            scores = {
                "stoi_noisy": results.get("stoi_noisy"),
                "stoi_enhanced": results.get("stoi_enhanced"),
                "stoi_delta": results.get("stoi_delta"),
                "files": results.get("files"),
                "pesq_noisy": results.get("pesq_noisy"),
                "pesq_enhanced": results.get("pesq_enhanced"),
            }

        ranked = sorted(rows, key=lambda r: r["enhanced"] - r["noisy"])
        best = [{"file": r["file"], "delta": r["enhanced"] - r["noisy"]}
                for r in ranked[-3:][::-1]]
        worst = [{"file": r["file"], "delta": r["enhanced"] - r["noisy"]}
                 for r in ranked[:3]]

        loss_png = _loss_curve(self.out_dir)
        charts = {
            "bars": _png_data_uri(_score_bars(scores)) if scores else None,
            "hist": _png_data_uri(_delta_hist(rows)) if rows else None,
            "scatter": _png_data_uri(_scatter(rows)) if rows else None,
            "loss": _png_data_uri(loss_png) if loss_png else None,
        }

        counts = self._counts()
        dataset_info = {
            "train_pairs": counts.get("clean_trainset_28spk_wav", 0),
            "val_pairs": counts.get("clean_val_wav", 0),
            "test_pairs": counts.get("clean_testset_wav", 0),
        }

        try:
            eval_mtime = os.path.getmtime(os.path.join(self.out_dir, "results.json"))
        except OSError:
            eval_mtime = None

        return {
            "has_results": scores is not None,
            "scores": scores,
            "config": config,
            "model": self.model_info(),
            "dataset": dataset_info,
            "charts": charts,
            "best": best,
            "worst": worst,
            "eval_mtime": eval_mtime,
        }
