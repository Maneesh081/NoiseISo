"""U-Net: a convolutional network shaped like the letter U.

Left leg (encoder): the spectrogram picture is repeatedly made smaller but deeper --
the model compresses it into a compact summary of "what is going on".
Bottom (bottleneck): the most compressed part.
Right leg (decoder): the compressed summary is expanded back to the original picture
size.
Skip connections: thin copies of the encoder's fine detail are carried straight across
the U so that small, easy-to-lose features (like faint speech harmonics) survive the
compression and come back on the right side.

The output is a MASK: a picture of the same size where every pixel is between 0 (pure
noise -> remove) and 1 (keep). We multiply the noisy picture by this mask to clean it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channels(base, depth):
    """Channel width at each U-Net level, from the top (finest) to the bottom."""
    return [base * (2 ** i) for i in range(depth)]


class DoubleConv(nn.Module):
    """Two conv3x3 + batch-norm + ReLU, with padding so the picture size is unchanged."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=16, depth=4):
        super().__init__()
        # widths[i] is the channel count of level i. widths[0] is the finest level.
        widths = _channels(base, depth)                 # e.g. [16, 32, 64, 128]

        # Encoder: level i takes in_ch (then the previous level) and emits widths[i].
        self.encoders = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.encoders.append(DoubleConv(prev, w))
            prev = w
        self.pool = nn.MaxPool2d(2)

        # Bottleneck doubles the deepest width once more.
        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)

        # Decoder walks back up. After the bottleneck we hold widths[-1]*2 channels;
        # each level upsamples, concatenates its skip (also widths[i]), and reduces to
        # widths[i]. So the input width of up level i is widths[i+1] (or *2 at the
        # bottleneck) and its output width is widths[i].
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        up_in = widths[-1] * 2
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(up_in, widths[i], 2, stride=2))
            self.decoders.append(DoubleConv(widths[i] * 2, widths[i]))  # up + skip
            up_in = widths[i]

        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            # Crop the 2x2-upsampled tensor to match the skip's size exactly.
            diff_t = skip.shape[-1] - h.shape[-1]
            diff_f = skip.shape[-2] - h.shape[-2]
            h = F.pad(h, [0, diff_t, 0, diff_f])
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return torch.sigmoid(self.head(h))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
