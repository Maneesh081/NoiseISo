"""PyTorch Dataset / DataLoader that turns noisy+clean spectrogram pairs into training
batches.

One pair = the same sentence spoken clean, and with noise added on top. The network sees
both pictures (the noisy one and the clean one) and learns to map one to the other.

Training chunks: we do not feed whole sentences. Instead we cut a random ~3-second segment
out of each ~4-second clip, so the model sees many different slices with a fixed time
length (keeps batches rectangular and GPU-friendly).
"""

import math
import os
import random

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from features import Spectrogram


class EnhancementDataset(Dataset):
    """VoiceBank+DEMAND style set. Expects clean_dir and noisy_dir to hold the SAME
    filenames; pairs are matched by shared basename.

    Returns (noisy_mag, clean_mag, scale):
        noisy_mag : (F, T)  magnitude spectrogram of the noisy chunk
        clean_mag : (F, T)  magnitude spectrogram of the clean chunk
        scale     : scalar, the peak value used to normalise the pair (so the audio
                    can be perfectly scaled back later)
    """

    def __init__(self, clean_dir, noisy_dir, stft: Spectrogram, sr=16000,
                 chunk_seconds=3.0, seed=0):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.stft = stft
        self.sr = sr
        self.chunk_samples = int(chunk_seconds * sr)
        self.rng = random.Random(seed)

        clean_names = {f for f in os.listdir(clean_dir) if f.endswith(".wav")}
        noisy_names = {f for f in os.listdir(noisy_dir) if f.endswith(".wav")}
        self.pairs = sorted(clean_names & noisy_names)
        if not self.pairs:
            raise RuntimeError(
                f"No matching wav pairs between {clean_dir} and {noisy_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def _read(self, path):
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != self.sr:
            raise RuntimeError(f"{path} is {sr} Hz, expected {self.sr} Hz.")
        x = torch.from_numpy(wav)
        if x.dim() > 1:  # safety: downmix to mono if stereo
            x = x.mean(dim=1)
        return x

    def __getitem__(self, index):
        name = self.pairs[index]
        clean = self._read(os.path.join(self.clean_dir, name))
        noisy = self._read(os.path.join(self.noisy_dir, name))

        # Both clips are the same recording, so use ONE normalisation factor so the
        # pair stays comparable.
        scale = noisy.abs().max().clamp_min(1e-6)
        clean = clean / scale
        noisy = noisy / scale

        # Make every sample exactly chunk_samples long. Longer clips get a random
        # window; shorter clips (plenty in VoiceBank) are zero-padded, otherwise
        # batches would contain mismatched spectrogram widths.
        n = min(clean.numel(), noisy.numel())
        clean = clean[:n]
        noisy = noisy[:n]
        chunk = self.chunk_samples
        if n >= chunk:
            start = self.rng.randrange(n - chunk + 1)
            clean = clean[start:start + chunk]
            noisy = noisy[start:start + chunk]
        elif n > 0:
            clean = F.pad(clean, (0, chunk - n))
            noisy = F.pad(noisy, (0, chunk - n))
        else:
            clean = torch.zeros(chunk)
            noisy = torch.zeros(chunk)

        clean_mag = self.stft.magnitude(clean.unsqueeze(0)).squeeze(0)    # (F, T)
        noisy_mag = self.stft.magnitude(noisy.unsqueeze(0)).squeeze(0)    # (F, T)
        return noisy_mag, clean_mag, torch.tensor(scale.item())


def make_loader(dataset, batch_size, shuffle=False, num_workers=0):
    def collate(items):
        noisy = torch.stack([i[0] for i in items])
        clean = torch.stack([i[1] for i in items])
        scale = torch.stack([i[2] for i in items])
        return noisy, clean, scale

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate,
    )


def frames_to_samples(frames, hop_length, center=True):
    """Convert a number of STFT frames back to audio samples (approximate)."""
    if center:
        return (frames - 1) * hop_length
    return frames * hop_length
