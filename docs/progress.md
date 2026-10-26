# Progress Log

Rule: this file is updated after every step with (1) what was done, (2) any errors and how they
were solved, (3) current status. No emojis - plain text only.

---

## 2026-09-11 - Session start

### Done
- Defined and locked the project plan for a real-time speech enhancement (noise suppression)
  U-Net, stored in DL.md.
- Locked UI spec: Gradio app with 2 tabs ("Enhance" and "Datasets"), file upload only
  (no browser mic), no emojis anywhere in the UI.

### Status
- Starting environment setup: Python venv + PyTorch (CUDA) + librosa etc.
- Dataset: VoiceBank + DEMAND (to be stored on /home/Guts/dl_data, large drive).

### Errors / fixes
- None yet.

## Build: features.py & dataset.py written
- src/features.py: STFT wrapper (magnitude/phase/reconstruct) + Spectrogram class.
- src/dataset.py: EnhancementDataset + collate, random 3s chunking, shared peak scale,
  paired clean/noisy matching by filename.
- Wrote src/model.py: U-Net (DoubleConv blocks, encoder pool, bottleneck, convT decoder,
  skip connections, sigmoid mask head).
### Errors / fixes
- First features.py draft had a buggy reconstruct helper; rewrote cleanly.
- dataset.py first draft was truncated/broken; fully rewritten.
### Status
- torch still downloading (cu128 wheels ~5GB). Code for features/dataset/model is ready.

## BIG FIX: disk layout (root cause of every stalled install)
- Symptom: torch/pip """hung""" then died with `[Errno 28] No space left on device`.
- Root cause: venv + data lived on C: (OneDrive, ~16GB free, 92% full). torch cu128
  wheels need ~6GB and the VoiceBank+DEMAND zips need ~3GB -> disk wall hit every time.
- Fix: moved the heavy stuff to D: (276G, 95G free), which WSL auto-mounts as /mnt/d:
    /mnt/d/DL_heavy/venv  <- .venv is a symlink here
    /mnt/d/DL_heavy/data  <- data is a symlink here
  The OneDrive project dir now carries only light code + docs (OneDrive keeps syncing
  them without choking on gigabytes of CUDA libs and wav files).
- Bonus: WSL mounts any drive plugged in; OneDrive-D itself backs up fine.

## 2026-09-23/24 - Project fully relocated to D:, environment completed, model trained

### Done
- Relocated the entire project to D: (local NTFS, not OneDrive). Verified the move file by
  file: all code and docs byte-identical (10/10 files). The OneDrive copy was removed, so
  D: is now the single source of truth.
- Consolidated the layout to one self-contained folder on D::
      /mnt/d/DL/            code, docs, .venv, data, out
  (earlier stray heavy dirs under /home/Guts were deleted).
- Built the venv on D: and bootstrapped pip 26.2.1 into it (the system Python has no
  ensurepip, so `python -m venv` alone cannot create a usable environment here).
- Installed torch 2.11.0+cu128 and torchaudio, CUDA verified working on the
  NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB).
- Installed the rest of the stack: numpy, scipy, soundfile, librosa, matplotlib, pillow,
  tensorboard, pystoi, gradio 6.28.0.
- Data: rather than the ~3.6 GB VoiceBank+DEMAND download, wrote src/make_synth_data.py,
  which generates paired clean/noisy clips in the exact VoiceBank directory layout
  (28 "speakers" x 4 utterances train, 4 x 4 test, 16 kHz, SNR 0/5/10/15/20 dB, ten
  DEMAND-style noise types). Deterministic and instant; 112 train pairs + 16 test pairs.
- Trained the U-Net: val loss 0.1783 -> 0.0339 in 6.7 min, train and val tracked together.
- Evaluated on the held-out test speakers: STOI 0.6550 -> 0.7335 (+0.0785) over 16 files.
- Launched the Gradio app: two tabs, verified serving over HTTP 200 with 49 components.

### Errors / fixes
- torch installs kept dying with `[Errno 28] No space left on device` partway through the
  820 MB wheel. Real cause: pip's unpack temp and wheel cache were landing on the
  near-full C: volume. Fixed by pinning TMPDIR, PIP_CACHE_DIR and XDG_CACHE_HOME to
  D: and running the install synchronously. Detached/background installs were also getting
  reaped mid-download, which is why several attempts produced a truncated file
  (a 140 MB "zip" that was not a valid zip at all).
- U-Net could not run at all: the first encoder was built as DoubleConv(16, 32) but
  received a 1-channel spectrogram, and the decoder channel bookkeeping did not line up
  (the first decoder emitted 128 channels where the next upconv expected 256). Rewrote
  the channel plan around widths[i] = base * 2**i and made the stem consume in_ch.
  Model is now 1.94 M parameters, inside the planned 2-4 M range.
- train.py had a duplicated `--out` argument (argparse conflict) and passed `args.amp`
  (an int) to autocast, which requires a bool. Fixed both.
- eval.py had `del set_seed` on a name that was never assigned locally -> UnboundLocalError.
  Removed the dead line. Also made PESQ optional: it is a C extension and this environment
  has no compiler, so eval now reports STOI and says PESQ is unavailable instead of crashing.
- While making PESQ optional the metric block was accidentally de-indented out of the
  per-file loop, so only the last file was scored (reported as "1 files"). Caught it
  because 16 wavs existed on disk; re-indented and confirmed 16/16 scored.
- app.py would not have run: Gradio 6 removed the `source=` argument of gr.Audio (now
  `sources=`), the spectrogram helper returned PNG bytes while the components declared
  type="pil", and the streaming path moved the mask to CPU before combining it with the
  CUDA phase. All three fixed.
- requirements.txt listed julius, tqdm and pyyaml, which no source file imports.
  Replaced with what the code actually uses.

## 2026-10-04 - UI improvements: run.sh, Model tab, My Tests tab, auto-resample

### Done
- Added run.sh, a single-command launcher: ./run.sh [app|train|eval|data|stop|help]. It
  resolves the project from its own location, so it works from any directory, checks the
  venv and checkpoint exist, and prints a concrete fix if either is missing.
- Added a Model tab showing the architecture table (1.94 M parameters, input/output shape,
  checkpoint size, GPU), the measured STOI before/after with the change, the training setup
  actually used, and a train/validation loss curve read from the TensorBoard event files.
  All values are read from out/results.json, out/config.json and the event files at page
  load, so they describe the checkpoint that is really loaded rather than fixed text.
- Added a My Tests tab backed by a new test_noisy/ folder. It lists every audio file it
  finds (wav, mp3, m4a, flac, ogg, aiff), and the chosen file can be enhanced with the
  same before/after spectrograms. Seeded the folder with four examples: traffic at 5 dB,
  babble at 10 dB, cafe at 0 dB, aircraft at 15 dB.
- Uploads are now auto-resampled to 16 kHz mono instead of being rejected, so 44.1 kHz and
  48 kHz recordings work directly. The tab reports the original rate when it converted.
- Fixed the header text so it no longer implies the training data is the recorded
  VoiceBank+DEMAND corpus; it now states plainly that the data is synthetic.

### Errors / fixes
- best_meta.json was not JSON: train.py called torch.save() on a .json filename, producing
  a binary zip archive that would crash any JSON reader. Now written with json.dump.
- The Model tab initially raised DuplicateBlockError because model_report() constructed
  Gradio components before entering the Blocks context, registering them once and then
  rendering them again. Changed it to return plain markdown strings plus at most one PIL
  image, with the components created inside the Blocks context.
- to_model_rate() was first defined nested inside build_app, where it could not be reused
  or tested directly; moved to module scope.
- Verifying the app with pgrep -f "app.py" gave a false positive, because the pattern
  matched the checking shell command itself. Switched to matching on the full python
  invocation and to an HTTP request, which is the real test.

### Status
- App is running at http://localhost:7860 with 4 tabs: Enhance, Datasets, Model, My Tests.
- PESQ is still unavailable in this environment (no C compiler); the Model tab shows STOI
  and labels PESQ as not available.
- PESQ is unavailable in this environment (no C compiler); install gcc, then
  `pip install pesq` to enable it.
- Training data is synthetic, so the numbers demonstrate the pipeline rather than
  benchmark-grade VoiceBank+DEMAND results. The real corpus can be dropped into
  data/ in the same directory layout to train on it without code changes.

## 2026-10-15 - Startup fix (venv moved to ext4), app fixes, real-data prep

### Done
- Diagnosed "cannot connect to localhost": nothing was listening because the app took
  minutes to boot. Root cause: the venv lived on /mnt/d, which WSL2 mounts as 9p; every
  module import stall. Measured `import gradio` at 1m37s (4.6s CPU, 96% I/O wait).
- Moved the venv to the native WSL ext4 filesystem (/home/Guts/dl-venv) and symlinked
  /mnt/d/DL/.venv -> that. Copy method matters: `cp -a` did ~15 files/s (gradio alone
  2m54s); `tar | tar` did the whole 7.5 GB / 42089 files in 21 min. Verified torch
  imports in 5s and gradio in 2.2s.
- run.sh now follows the symlink to the real path before invoking python. Running python
  *through* the /mnt/d/.venv symlink kept sys.prefix on 9p and re-slowed imports to 21s,
  so the resolved /home/guts/dl-venv path must be used.
- Confirmed the app now answers HTTP 200 within ~50 s (previously 2-4 min), so the
  browser no longer gives up. Removed the old 7.5 GB venv copy (freed ~7 GB on D:).
- Fixed the My Tests tab crash found in the old app.log: list_test_files returns bare
  filenames and enhance_file opened them relative to src/, causing a LibsndfileError.
  enhance_file now joins the name against test_dir.
- Bound the app to 0.0.0.0 (new --host arg, default 0.0.0.0) so the Windows browser can
  reach it; previously 127.0.0.1 only accepted connections from inside WSL.
- Added the missing README.md that DL.md and run.sh referred to.
- Real dataset: the existing DS_10283_2791.zip (1.09 GB) was corrupt (BadZipFile). The
  four official Edinburgh DataShare zips were located and a chunked downloader written,
  but the user opted to download the corpus themselves. Partial downloads and the corrupt
  zip were removed.
- Added src/prepare_real_data.py: finds the four sets anywhere under a staging dir
  (auto-extracting individual or combined zips, including archives inside archives),
  resamples to 16 kHz mono, and writes the project layout preserving filenames so clean
  and noisy pairs still match. Exposed as `./run.sh prepare [DIR]` (default data/_raw).
  Tested on synthetic (16 kHz pass-through) and a synthetic 48 kHz zip (resampled).

### Errors / fixes
- First `cp -a` of the venv was killed by the 25 min tool timeout after 2.3 GB / 28k
  files; switched to a tar stream, which was ~12x faster.
- `pkill -f download_vb.sh` matched the shell command running it and killed the shell
  (no output, apparent hang). Killed by PID instead.
- An `ls`/`pgrep` while the 9p rm and downloads were active hung the shell; avoided heavy
  directory scans on /mnt/d while busy.

### Status
- App works and starts in under a minute; venv fast on ext4; README added.
- Waiting on the real VoiceBank+DEMAND corpus: drop the zips (or extracted folders) in
  D:\DL\data\_raw and run `./run.sh prepare`, or pass another path. Then retrain and
  re-evaluate. Training data is still the synthetic corpus for now.

## 2026-10-26 - Real dataset, task scripts, FastAPI dashboard

### Done
- Resampled the real 48 kHz VoiceBank+DEMAND corpus to 16 kHz with parallel
  workers (`./prepare.sh`, 6 workers). Wrote 24,792 files to `data/`; the 48 kHz
  originals remain untouched in `data/_raw/`.
- Speaker-disjoint validation split: p286 and p287 held out (872 clips). Training
  has 10,700 clips from 26 speakers; the official 824-clip test set is only used
  by evaluation.
- Split the old `run.sh` subcommands into task scripts: `run.sh` (dashboard),
  `train.sh`, `eval.sh`, `prepare.sh`, `synth.sh`, `stop.sh`, `help.sh`, sharing
  `common.sh`.
- `train.py`: added `--clean_val/--noisy_val` (falls back to the test set),
  default patience 8, and kept the dataset STFT on CPU so DataLoader workers are
  safe.
- `dataset.py`: real clips have mixed durations; every sample is now cropped or
  zero-padded to exactly the 3 s chunk so batches stack.
- Added `src/engine.py` (checkpoint loading, overlap-add inference, spectrogram/
  waveform/chart rendering) and rewrote `src/app.py` as FastAPI.
- Added `src/web/`: dark-theme dashboard with Overview, Dataset, Model, Enhance
  and Live tabs (record-then-clean with a live spectrogram).
- `requirements.txt`: removed Gradio, added FastAPI and Uvicorn.
- Removed the partial ext4 duplicate left by an aborted dataset move; verified the
  D: originals were complete first.

### Errors / fixes
- An interrupted cross-filesystem `mv` of `data/_raw` left a partial copy on
  ext4. Verified all four source folders still had full counts, then deleted the
  duplicate and kept everything on D: as requested.
- First real-data training run crashed in the DataLoader: short clips produced
  spectrograms of different widths (257x376 vs 257x246). Fixed by padding/cropping
  every sample to the fixed chunk length.
- Moved the dataset's STFT window off CUDA after adding worker processes, to avoid
  forking a CUDA tensor.
- `train.sh` now clears `out/tb` at the start so a failed run cannot leave an
  empty event file that hides the real loss curve.
- The 9p mount on D: makes per-file reads slow; training is I/O bound. Epoch 1
  took ~408 s, so a full run is several hours rather than the ~1.5 h seen on the
  synthetic corpus.

### Status
- Training on the real corpus is running. Epoch 1: train 0.0783, val 0.0682.
- FastAPI endpoints smoke-tested against the previous synthetic checkpoint:
  `/`, `/api/health`, `/api/overview`, `/api/dataset`, `/api/audio`, `/api/spec`,
  `/api/demo` and `POST /api/enhance` all respond correctly.
- Remaining: finish training, run `./eval.sh`, restart the dashboard on the new
  checkpoint, verify the real score charts and the microphone tab.
