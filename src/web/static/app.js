/* Dashboard behaviour: tabs, overview charts, dataset browser, upload enhance,
   and the microphone recorder. No frameworks. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = { overview: null, datasetLoaded: false };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

const fmt = (v, digits = 3) =>
  (v === null || v === undefined || Number.isNaN(Number(v))) ? "\u2014" : Number(v).toFixed(digits);

const signed = (v, digits = 3) =>
  (v === null || v === undefined) ? "\u2014" : (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(digits);

const rmsDb = (v) => (v > 0 ? (20 * Math.log10(v)).toFixed(1) : "-inf");

const baseName = (name) => String(name || "audio").replace(/\.[^.]+$/, "");

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let msg = "HTTP " + r.status;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ------------------------------------------------------------------ tabs */
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => activate(btn.dataset.tab));
});

function activate(name, updateHash = true) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (updateHash) history.replaceState(null, "", "#" + name);
  if (name === "dataset" && !state.datasetLoaded) loadDataset();
  if (name === "model" && state.overview) renderModel(state.overview);
}

/* -------------------------------------------------------------- overview */
function renderOverview(o) {
  state.overview = o;
  const s = o.scores;

  const cards = [
    { label: "STOI \u00b7 noisy", value: s ? fmt(s.stoi_noisy) : "\u2014", sub: "intelligibility", cls: "" },
    { label: "STOI \u00b7 enhanced", value: s ? fmt(s.stoi_enhanced) : "\u2014", sub: "after the model", cls: "good" },
    { label: "Change", value: s ? signed(s.stoi_delta) : "\u2014", sub: "higher is better", cls: "accent" },
    { label: "Test clips", value: s ? String(s.files) : "\u2014", sub: "official test set", cls: "violet" },
  ];
  $("#kpis").innerHTML = cards.map((c) => `
    <div class="kpi ${c.cls}">
      <span class="k-label">${esc(c.label)}</span>
      <span class="k-value">${esc(c.value)}</span>
      <span class="k-sub">${esc(c.sub)}</span>
    </div>`).join("");

  $("#overview-note").textContent = o.has_results
    ? `Averaged over ${s.files} held-out clips. STOI runs from 0 to 1; higher means more intelligible.`
    : "No evaluation yet. Run ./eval.sh to score the checkpoint on the official test set.";

  const figures = [
    ["bars", "STOI: noisy vs enhanced"],
    ["loss", "Training progress"],
    ["hist", "Per-file STOI change"],
    ["scatter", "Before vs after, per file"],
  ].filter(([key]) => o.charts[key]);

  $("#overview-charts").innerHTML = figures.length
    ? figures.map(([key, caption]) => `
        <figure>
          <figcaption>${esc(caption)}</figcaption>
          <img src="${o.charts[key]}" alt="${esc(caption)}">
        </figure>`).join("")
    : `<div class="card"><p class="muted">Charts appear once a trained checkpoint has been evaluated.</p></div>`;

  const fileList = (title, items, positive) => {
    if (!items || !items.length) return "";
    return `
      <div class="card">
        <h3>${esc(title)}</h3>
        <ul class="list">
          ${items.map((f) => `
            <li>
              <span class="file">${esc(f.file)}</span>
              <span class="${f.delta >= 0 ? "delta-pos" : "delta-neg"}">${signed(f.delta)}</span>
            </li>`).join("")}
        </ul>
      </div>`;
  };

  $("#overview-files").innerHTML =
    fileList("Biggest improvements", o.best) +
    fileList("Hardest clips", o.worst);
}

/* ----------------------------------------------------------------- model */
function renderModel(o) {
  const m = o.model;
  const stats = [
    ["Parameters", m.params_m.toFixed(2) + " M"],
    ["Input", `1 \u00d7 ${m.freq_bins} \u00d7 frames`],
    ["Output", `1 \u00d7 ${m.freq_bins} \u00d7 frames mask`],
    ["Sample rate", m.sr + " Hz"],
    ["STFT", `${m.n_fft} / hop ${m.hop}`],
    ["Checkpoint", m.checkpoint_mb.toFixed(1) + " MB"],
    ["Runs on", m.gpu],
  ];
  $("#model-stats").innerHTML = stats.map(([k, v]) => `
    <div class="stat">
      <span class="s-label">${esc(k)}</span>
      <span class="s-value" style="font-size:1.02rem">${esc(v)}</span>
    </div>`).join("");

  const c = o.config || {};
  const rows = [
    ["Epochs", c.epochs ?? "\u2014"],
    ["Best validation loss", c.best_val !== undefined ? Number(c.best_val).toFixed(4) : "\u2014"],
    ["Batch size", c.batch_size ?? "\u2014"],
    ["Learning rate", c.lr ?? "\u2014"],
    ["Chunk length", (c.chunk_seconds ?? "\u2014") + (c.chunk_seconds ? " s" : "")],
    ["Base channels / depth", `${c.base ?? "\u2014"} / ${c.depth ?? "\u2014"}`],
    ["Precision", c.amp ? "FP16 autocast" : "FP32"],
  ];
  $("#model-config").innerHTML = rows.map(([k, v]) => `
    <span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`).join("");

  const loss = $("#model-loss");
  const fig = loss.closest("figure");
  if (o.charts.loss) {
    loss.src = o.charts.loss;
    fig.classList.remove("hidden");
  } else {
    fig.classList.add("hidden");
  }
}

/* --------------------------------------------------------------- dataset */
async function loadDataset() {
  try {
    const d = await getJSON("/api/dataset");
    renderDataset(d);
    state.datasetLoaded = true;
  } catch (e) {
    $("#dataset-stats").innerHTML =
      `<div class="card"><p class="muted">Could not load dataset: ${esc(e.message)}</p></div>`;
  }
}

function renderDataset(d) {
  const stats = [
    ["Training pairs", d.train_pairs, "26 speakers"],
    ["Validation pairs", d.val_pairs, "held-out speakers"],
    ["Test pairs", d.test_pairs, "official test set"],
    ["Speakers", d.num_speakers, "in the training split"],
  ];
  $("#dataset-stats").innerHTML = stats.map(([label, value, sub]) => `
    <div class="stat">
      <span class="s-label">${esc(label)}</span>
      <span class="s-value">${Number(value).toLocaleString()}</span>
      <span class="s-sub">${esc(sub)}</span>
    </div>`).join("");

  $("#dataset-snr").innerHTML = d.snr_db.map((v) => `<span class="chip accent">${v} dB</span>`).join("");
  $("#dataset-noise").innerHTML = d.noise_types.map((n) => `<span class="chip">${esc(n)}</span>`).join("");

  const host = $("#dataset-samples");
  if (!d.samples.length) {
    host.innerHTML = `<div class="card"><p class="muted">Converted 16 kHz test clips were not found.</p></div>`;
    return;
  }
  host.innerHTML = d.samples.map((s) => `
    <div class="sample">
      <div class="sample-head">
        <span class="sample-name">${esc(s.name)}</span>
      </div>
      <div class="sample-grid">
        <div class="sample-col">
          <div class="col-head">Noisy</div>
          <audio controls preload="none" src="${s.noisy_audio}"></audio>
          <img src="${s.noisy_spec}" alt="Noisy spectrogram" loading="lazy">
        </div>
        <div class="sample-col">
          <div class="col-head">Clean original</div>
          <audio controls preload="none" src="${s.clean_audio}"></audio>
          <img src="${s.clean_spec}" alt="Clean spectrogram" loading="lazy">
        </div>
      </div>
    </div>`).join("");
}

/* --------------------------------------------------------------- enhance */
const enhanceStatus = $("#enhance-status");
const enhanceResult = $("#enhance-result");
const fileInput = $("#file-input");
const dropzone = $("#dropzone");

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (f) enhanceBlob(f, f.name);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) enhanceBlob(f, f.name);
});

function resultHTML(res, name) {
  const st = res.stats;
  const note = res.resampled
    ? `Input ${res.original_sr} Hz \u2192 16 kHz`
    : `Input ${res.original_sr} Hz`;
  return `
    <div class="result-grid">
      <div class="card">
        <h3>Noisy input</h3>
        <audio controls src="${res.noisy_audio}"></audio>
        <img src="${res.noisy_spec}" alt="Noisy spectrogram">
      </div>
      <div class="card">
        <h3>Enhanced output</h3>
        <audio controls src="${res.clean_audio}"></audio>
        <img src="${res.clean_spec}" alt="Enhanced spectrogram">
        <a class="btn" download="${esc(baseName(name))}-enhanced.wav" href="${res.clean_audio}">Download WAV</a>
      </div>
    </div>
    <figure class="card wide">
      <figcaption>Waveform \u00b7 noisy (grey) and cleaned (blue)</figcaption>
      <img src="${res.waveform}" alt="Waveforms">
    </figure>
    <p class="muted">${res.duration.toFixed(2)} s \u00b7 ${note} \u00b7
      RMS ${rmsDb(st.rms_in)} dB \u2192 ${rmsDb(st.rms_out)} dB</p>`;
}

async function enhanceBlob(blob, name) {
  enhanceResult.classList.add("hidden");
  enhanceStatus.className = "status-line busy";
  enhanceStatus.innerHTML = `<span class="spinner"></span>Enhancing \u201c${esc(name)}\u201d\u2026`;
  try {
    const r = await fetch("/api/enhance", {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
    });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const res = await r.json();
    enhanceResult.innerHTML = resultHTML(res, name);
    enhanceResult.classList.remove("hidden");
    enhanceStatus.className = "status-line";
    enhanceStatus.textContent = "Done.";
    enhanceResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    enhanceStatus.className = "status-line err";
    enhanceStatus.textContent = "Error: " + e.message;
  }
}

async function loadDemo() {
  try {
    const d = await getJSON("/api/demo");
    const host = $("#demo-list");
    if (!d.files.length) {
      host.innerHTML = `<span class="demo-empty">No clips in test_noisy/.</span>`;
      return;
    }
    host.innerHTML = d.files.map((f) => `
      <button class="demo-item" data-name="${esc(f.name)}">
        <span>${esc(f.name)}</span><span class="go">enhance</span>
      </button>`).join("");
    host.querySelectorAll(".demo-item").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const name = btn.dataset.name;
        try {
          const r = await fetch("/api/audio?set=test_noisy&name=" + encodeURIComponent(name));
          if (!r.ok) throw new Error("could not load " + name);
          await enhanceBlob(await r.blob(), name);
        } catch (e) {
          enhanceStatus.className = "status-line err";
          enhanceStatus.textContent = "Error: " + e.message;
        }
      });
    });
  } catch (_) {
    $("#demo-list").innerHTML = `<span class="demo-empty">Could not list clips.</span>`;
  }
}

/* ------------------------------------------------------------------ live */
const live = {
  running: false, stream: null, ctx: null, source: null, analyser: null,
  proc: null, mute: null, chunks: [], raf: 0, timer: 0, start: 0,
};

const liveBtn = $("#live-btn");
const liveCanvas = $("#live-canvas");
const liveStatus = $("#live-status");
const liveResult = $("#live-result");
const liveCtx = liveCanvas.getContext("2d");

function resetCanvas() {
  liveCtx.fillStyle = "#080b10";
  liveCtx.fillRect(0, 0, liveCanvas.width, liveCanvas.height);
}
resetCanvas();

liveBtn.addEventListener("click", () => (live.running ? stopLive() : startLive()));

async function startLive() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    liveStatus.className = "status-line err";
    liveStatus.textContent = "This browser does not expose a microphone here. Use http://localhost:7860.";
    return;
  }
  liveResult.classList.add("hidden");
  try {
    live.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (e) {
    liveStatus.className = "status-line err";
    liveStatus.textContent = "Microphone error: " + e.message;
    return;
  }

  live.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (live.ctx.resume) await live.ctx.resume();
  live.source = live.ctx.createMediaStreamSource(live.stream);
  live.analyser = live.ctx.createAnalyser();
  live.analyser.fftSize = 2048;
  live.analyser.smoothingTimeConstant = 0.78;
  live.source.connect(live.analyser);

  live.proc = live.ctx.createScriptProcessor(4096, 1, 1);
  live.chunks = [];
  live.proc.onaudioprocess = (e) => {
    if (!live.running) return;
    live.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  live.mute = live.ctx.createGain();
  live.mute.gain.value = 0;
  live.source.connect(live.proc);
  live.proc.connect(live.mute);
  live.mute.connect(live.ctx.destination);

  live.running = true;
  live.start = performance.now();
  liveBtn.textContent = "Stop and clean";
  liveBtn.classList.add("recording");
  $("#live-hint").textContent = `Recording at ${live.ctx.sampleRate} Hz`;
  liveStatus.className = "status-line busy";
  liveStatus.textContent = "Recording\u2026 press stop when finished.";
  resetCanvas();
  drawLiveFrame();
  updateLiveTimer();
  live.timer = setInterval(updateLiveTimer, 100);
}

function updateLiveTimer() {
  $("#live-timer").textContent = ((performance.now() - live.start) / 1000).toFixed(1) + " s";
}

function heat(v) {
  const r = Math.round(255 * Math.min(1, v * 2.2));
  const g = Math.round(255 * Math.min(1, v * 1.25));
  const b = Math.round(255 * Math.min(1, 0.12 + v * 0.95));
  return `rgb(${r},${g},${b})`;
}

function drawLiveFrame() {
  if (!live.running) return;
  const w = liveCanvas.width;
  const h = liveCanvas.height;
  liveCtx.drawImage(liveCanvas, -2, 0);
  const bins = new Uint8Array(live.analyser.frequencyBinCount);
  live.analyser.getByteFrequencyData(bins);
  for (let y = 0; y < h; y++) {
    const idx = Math.floor((1 - y / h) * (bins.length - 1));
    liveCtx.fillStyle = heat(bins[idx] / 255);
    liveCtx.fillRect(w - 2, y, 2, 1);
  }
  live.raf = requestAnimationFrame(drawLiveFrame);
}

function encodeWAV(chunks, sampleRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, length * 2, true);
  let off = 44;
  for (const c of chunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
  }
  return new Blob([view], { type: "audio/wav" });
}

async function stopLive() {
  if (!live.running) return;
  live.running = false;
  clearInterval(live.timer);
  cancelAnimationFrame(live.raf);
  const sr = live.ctx.sampleRate;
  try { live.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
  try {
    live.proc.disconnect();
    live.mute.disconnect();
    live.source.disconnect();
  } catch (_) {}
  try { await live.ctx.close(); } catch (_) {}
  liveBtn.textContent = "Start recording";
  liveBtn.classList.remove("recording");

  const total = live.chunks.reduce((n, c) => n + c.length, 0);
  if (!total) {
    liveStatus.className = "status-line err";
    liveStatus.textContent = "No audio was captured. Try again and speak while recording.";
    return;
  }

  liveStatus.className = "status-line busy";
  liveStatus.innerHTML =
    `<span class="spinner"></span>Cleaning ${(total / sr).toFixed(1)} s of audio\u2026`;
  const blob = encodeWAV(live.chunks, sr);
  try {
    const r = await fetch("/api/live", {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: blob,
    });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const res = await r.json();
    liveResult.innerHTML = resultHTML(res, "microphone");
    liveResult.classList.remove("hidden");
    liveStatus.className = "status-line";
    liveStatus.textContent = `Done \u00b7 ${(total / sr).toFixed(1)} s recorded at ${sr} Hz.`;
    liveResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    liveStatus.className = "status-line err";
    liveStatus.textContent = "Error: " + e.message;
  }
  live.chunks = [];
}

/* ------------------------------------------------------------------ init */
async function init() {
  try {
    const o = await getJSON("/api/overview");
    renderOverview(o);
    renderModel(o);
    const st = $("#status");
    st.textContent = `${o.model.device} \u00b7 ${o.model.params_m.toFixed(2)} M params`;
    st.className = "status ok";
  } catch (e) {
    const st = $("#status");
    st.textContent = "backend error";
    st.className = "status err";
    $("#overview-note").textContent = e.message;
  }
  loadDemo();
  const h = location.hash.replace("#", "");
  if (h && document.getElementById(h)) activate(h, false);
}

init();
