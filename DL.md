# Real-Time Noise Suppression (Speech Enhancement)

A Deep Learning system that removes unwanted background noise (fan, keyboard, mouse, traffic,
AC hum, background chatter, etc.) from microphone audio in real time. The cleaned audio is
suitable for Discord, online meetings, gaming, and voice calls.

```
Microphone Input                          Cleaned Output
(Your voice + fan + keyboard + traffic)
        |                                        ^
        v                                        |
   Spectrogram  ->  U-Net  ->  Mask  ->  Cleaned Spectrogram
```

---

## How it works (plain-English)

Before we get to the code, here are the handful of ideas this project is built on.
If you understand these, you understand 90% of the project.

### 1. What is a spectrogram?
Sound is just air vibrating. A microphone turns that vibration into a **waveform** —
a line that goes up and down (amplitude over time).

But it's hard for a neural network to learn from a raw waveform directly. So instead we
convert it into a **spectrogram**: a picture of sound, where
- the **horizontal axis** is time,
- the **vertical axis** is frequency (low = bass, high = treble),
- the **color** shows how loud each frequency is at each moment.

A spectrogram literally looks like a grayscale image. When someone speaks you see bright
vertical stripes (their voice comes in short bursts). Background noise fills in the in-between
areas. **That's why we can "see" the noise as extra brightness** in the picture.

```
freq ^        "noise haze" (fan/keyboard) fills this area
     |   █ █   █ █     <- voice stripes (bright, structured)
     |  █ █ █ █ █ █
     |` ` ` ` ` ` ` ` <- scattered noise dots
     +--------------------->
             time
```

### 2. What is STFT? (converts audio → spectrogram)
**Short-Time Fourier Transform.** The audio is chopped into small overlapping windows
(e.g. 32 ms each). Each window's frequency content is computed, and the results are stacked
side-by-side to form the spectrogram. We'll use PyTorch built-ins, so no math needed from you —
but now you know the name.

### 3. What is a "mask"?
The trick that makes this whole project work: instead of generating audio from nothing,
we ask the model "for every little square in the spectrogram picture, **how much** of this
is voice vs. noise?"

The model outputs a **mask**: a grid of numbers between 0 and 1, one per square.
- 1.0 = keep this (it's your voice)
- 0.0 = remove this (it's noise)
- 0.5 = partly keep

We then **multiply** the noisy picture by the mask. Squares that are pure noise get zeroed out.
The voice stays.

```
Noisy picture  ×  Mask  =  Clean picture
   (has noise)     (0s and 1s)      (voice only)
```

### 4. What is a U-Net?
Our model. It's a type of **CNN** (Convolutional Neural Network — the same family of model
used for image tasks like medical scans or photos). "U-Net" is named after its shape, which
looks like the letter **U**:

- The **left side (going down)** = the model shrinks the picture, learning "what is here"
  (compressing it into a small summary of features).
- The **bottom** = the most compressed summary.
- The **right side (going up)** = the model re-expands the picture back to full size,
  turning its understanding back into a full-size prediction.
- **Skip connections** (the bridges going straight across) carry *fine detail* from the
  left to the right, so small noise details don't get lost.

```
        INPUT (noisy picture)                 OUTPUT (mask)  = same size
                 │                                   ▲
        ┌────────▼────────┐             ┌────────────┘
        │  shrink (encode)│─── skip ───▶│  expand (decode)
        │  deeper, smaller│             │  back to full size
        └────────┬────────┘             └────────────┘
                 └──────▶  summary ◀──────┘
```

It's a small one for us (~2–4 million parameters) so it trains fast on your laptop GPU.

### 5. What are PESQ and STOI? (how we score quality)
Numbers that rate how good the cleaned audio is, compared to the original clean recording:

- **PESQ** (Perceptual Evaluation of Speech Quality) — score from -0.5 to 4.5. Higher is better.
  Models whether it *sounds* good to a human ear (4.0 ≈ perfect).
- **STOI** (Short-Time Objective Intelligibility) — score from 0 to 1. Higher is better.
  Measures how *understandable* the words are (0.9+ ≈ very clear).

These are the standard numbers people compare speech-enhancement systems on. Our target is
roughly the ballpark of published results for this dataset (~PESQ 2.8+, STOI 0.95+).

### 6. What does "real time" mean here?
Audio is processed in small **chunks** as they arrive, not the whole file at once. Our model
is small and fast, so cleaning is much faster than real time (a real-time-factor below 1 =
faster than the audio plays). This is what allows live Discord-style use.

### 7. What does the project actually contain?
| Piece | What it does | Where |
|---|---|---|
| Spectrogram converter | turns audio into the "picture" | `src/features.py` |
| Dataset loader | reads the training files | `src/dataset.py` |
| U-Net model | the brain that makes the mask | `src/model.py` |
| Training loop | teaches the model with data | `src/train.py` |
| Evaluation | scores quality (PESQ/STOI) | `src/eval.py` |
| Inference engine | cleans audio in chunks | `src/infer.py` |
| **User interface** | **upload a noisy file → get clean audio + see before/after pictures** | `src/app.py` |

---

## Training data: VoiceBank + DEMAND

- Clean human speech + the same speech with real background noise mixed in (~3 GB, the standard
  research dataset for this task).
- We add extra noise files (fan, keyboard, mouse, etc.) so those specific sounds get suppressed
  reliably — this is what makes *your* test audio work.
- All audio is resampled to 16 kHz (good enough for speech, small enough to train fast).

### Will it remove my fan / keyboard?
It removes whatever noise **types it was trained on**. VoiceBank+DEMAND covers ambient,
mechanical, and street-type noise. We deliberately add **fan, keyboard, and mouse noise files**
during training and then test with your own recording to confirm those specific ones disappear.

---

## Model

Lightweight **U-Net** operating on the spectrogram (magnitude) picture (~2–4 M params).
Fits comfortably in the 6 GB VRAM of the RTX 4050, fast enough for real-time inference.

## Training

- **Loss:** compares the model's cleaned picture against the true clean picture (L1 distance).
- Adam optimizer + cosine learning rate schedule, mixed precision (fp16), early stopping.
- Logs loss + sample spectrograms to TensorBoard so we can *watch* it learn.
- A few hours of training on the laptop GPU.

## Evaluation

Objective scoring with **PESQ** and **STOI** over all 824 held-out test clips,
plus a per-file results table (CSV) so we can find which noise types are weakest.

## User Interface (Gradio)

- **Upload a noisy audio file** (or record straight from the browser mic) → get the **cleaned
  audio** back, plus a download button.
- See **before vs. after** side-by-side: waveforms, spectrograms, and mel-spectrograms
  (audio "pictures" rendered in the browser).
- Built-in demo files to click and try instantly.
- Launch: `python src/app.py` then open the link in your browser.

---

## Project layout

```
DL/
├── DL.md                       # this file (plan + guide)
├── README.md                   # quickstart + how-it-works guide
├── requirements.txt
├── src/
│   ├── features.py             # STFT / spectrogram conversion
│   ├── dataset.py              # loading VoiceBank + DEMAND
│   ├── model.py                # U-Net definition
│   ├── train.py                # training loop
│   ├── eval.py                 # PESQ / STOI evaluation
│   ├── infer.py                # chunked (streaming) inference
│   └── app.py                  # Gradio user interface
├── checkpoints/                # trained model weights
└── output/                     # cleaned audio demos
```

## Milestones

1. Environment setup (Python venv, PyTorch with GPU support)
2. Download + prepare VoiceBank + DEMAND data
3. Build spectrogram features + dataset loader
4. Build and train the U-Net (a few hours)
5. Evaluate with PESQ / STOI
6. Build the Gradio UI with before/after visualizations
7. Test with your own noisy audio file
