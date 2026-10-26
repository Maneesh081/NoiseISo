# Real-Time Noise Suppression (Speech Enhancement)

A small U-Net that removes background noise (traffic, cafe, keyboard, chatter) from
speech. The cleaned audio is suitable for calls, meetings and voice chat.

See `DL.md` for the plain-English explanation of how it works. This file is the
quickstart.

## What is in the box

- **Model**: U-Net (`src/model.py`) that predicts a 0-1 mask over the noisy
  spectrogram; the mask is multiplied into the noisy magnitude and the original
  noisy phase is reused to synthesize audio.
- **Training**: paired clean/noisy clips, random 3-second chunks, L1 loss, AMP,
  early stopping (`src/train.py`).
- **Evaluation**: STOI on the official test set (`src/eval.py`).
- **Dashboard**: FastAPI backend + hand-written HTML/CSS/JS (`src/app.py`,
  `src/engine.py`, `src/web/`). No Gradio.
- **Data prep**: resamples the 48 kHz VoiceBank+DEMAND corpus to 16 kHz and creates
  a speaker-disjoint validation split (`src/prepare_real_data.py`).

## Layout

```
DL/
├── DL.md                    # plan + how-it-works guide
├── progress.md              # build log
├── common.sh                # shared env for the task scripts (sourced)
├── run.sh                   # start the dashboard
├── train.sh                 # train the U-Net
├── eval.sh                  # score the checkpoint (STOI)
├── prepare.sh               # resample the real dataset + val split
├── synth.sh                 # generate the synthetic fallback corpus
├── stop.sh                  # stop a running dashboard
├── help.sh                  # list the scripts
├── requirements.txt
├── src/
│   ├── features.py          # STFT / spectrogram conversion
│   ├── dataset.py           # paired clean/noisy loader (fixed 3 s chunks)
│   ├── model.py             # U-Net definition
│   ├── train.py             # training loop
│   ├── eval.py              # STOI (+ PESQ if installed)
│   ├── engine.py            # inference + plots used by the web app
│   ├── app.py               # FastAPI server
│   ├── prepare_real_data.py # resample real VoiceBank+DEMAND to 16 kHz
│   ├── make_synth_data.py   # synthetic corpus generator
│   └── web/                 # index.html + style.css + app.js
├── data/                    # 16 kHz clean/noisy sets
│   └── _raw/                # original 48 kHz corpus (untouched backup)
├── out/                     # checkpoint, results, TensorBoard logs
├── test_noisy/              # your own clips for the Enhance tab
└── .venv -> /home/Guts/dl-venv   # venv on ext4 for fast imports
```

## Environment

Python 3.11 with PyTorch (CUDA) and FastAPI. The venv lives on the native WSL
ext4 filesystem (`/home/Guts/dl-venv`) and is symlinked as `.venv`, because
importing Torch/FastAPI from the slow `/mnt/d` 9p mount takes minutes instead of
seconds. The task scripts follow the symlink to the real path automatically.

To rebuild the venv:

```bash
python3 -m venv --without-pip /home/Guts/dl-venv
PYTHONPATH="$HOME/.local/lib/python3.11/site-packages" \
  /home/Guts/dl-venv/bin/python -m pip install pip
ln -sfn /home/Guts/dl-venv /mnt/d/DL/.venv
# CUDA wheels come from the PyTorch index, not PyPI:
/mnt/d/DL/.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
/mnt/d/DL/.venv/bin/python -m pip install -r requirements.txt
```

## Data

The real VoiceBank+DEMAND corpus is ~48 kHz; the model works at 16 kHz, so it is
resampled once. Original 48 kHz files are kept in `data/_raw/` as a backup.

```bash
./prepare.sh                       # data/_raw -> 16 kHz sets in data/
./prepare.sh /path/to/downloads    # or another staging folder
```

Resulting 16 kHz layout:

```
data/clean_trainset_28spk_wav   10,700 clips (26 speakers)
data/noisy_trainset_28spk_wav   10,700 clips
data/clean_val_wav                 872 clips (p286, p287 held out)
data/noisy_val_wav                 872 clips
data/clean_testset_wav             824 clips (official test speakers)
data/noisy_testset_wav             824 clips
```

The validation speakers are excluded from training, so validation loss measures
generalization. The official test set is only used by `./eval.sh`.

Synthetic fallback (no download): `./synth.sh` (refuses unless `--force`).

## Usage

```bash
./run.sh            # start the dashboard, open http://localhost:7860
./train.sh          # train (40 epochs, early stopping, patience 8)
./eval.sh           # score the checkpoint on the official test set
./prepare.sh        # resample the real corpus + validation split

./stop.sh           # stop a running dashboard

```



## Dashboard

Five tabs:

- **Overview** - STOI before/after, per-file improvement distribution, training
  loss, and the best/hardest test clips.
- **Dataset** - dataset sizes, SNR levels, noise types, and sample clean/noisy
  pairs with audio players and spectrograms.
- **Model** - U-Net architecture diagram, parameter count, STFT settings, training
  config, and loss curve.
- **Enhance** - upload a clip (wav/flac/ogg/mp3/m4a) or pick one from
  `test_noisy/`; shows noisy vs cleaned audio, spectrograms, waveform, and a
  download link.
- **Live** - record from the microphone with a live spectrogram preview, then clean
  it. Browser echo cancellation / noise suppression / auto gain are disabled so the
  model receives the raw signal.

The live tab needs `http://localhost:7860` (a secure context); the LAN IP will not
be allowed to open the microphone by the browser.


## Metrics

- **STOI** (intelligibility, 0..1, higher is better) is always reported.

