"use strict";

// DRISHTI swipe viewer — self-contained, no external deps, runs from file://.
// Expects assets/ produced by scripts/make_preview.py.

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const pan = $("pan");
const after = $("after");
const before = $("before");
const beforeWrap = $("before-wrap");
const divider = $("divider");
const handle = $("handle");

const view = { x: 0, y: 0, scale: 1 }; // pan translate (screen px) + zoom
let img = { w: 0, h: 0 };
let split = 0.5; // fraction of image width where the swipe sits

const SETS = {
  rgb: { before: "assets/lr_rgb.png", after: "assets/sr_rgb.png" },
  nir: { before: "assets/lr_nir.png", after: "assets/sr_nir.png" },
};

function applyTransform() {
  pan.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
}

function applySplit() {
  const px = split * img.w;
  beforeWrap.style.width = px + "px";
  divider.style.left = px + "px";
}

function fit() {
  if (!img.w) return;
  const pad = 24;
  const s = Math.min(
    (stage.clientWidth - pad) / img.w,
    (stage.clientHeight - pad) / img.h
  );
  view.scale = s;
  view.x = (stage.clientWidth - img.w * s) / 2;
  view.y = (stage.clientHeight - img.h * s) / 2;
  applyTransform();
}

function loadSet(key) {
  const set = SETS[key];
  return new Promise((resolve) => {
    let n = 0;
    const done = () => {
      if (++n === 2) {
        img = { w: after.naturalWidth, h: after.naturalHeight };
        pan.style.width = img.w + "px";
        pan.style.height = img.h + "px";
        before.style.width = img.w + "px";
        after.style.width = img.w + "px";
        applySplit();
        resolve();
      }
    };
    after.onload = done;
    before.onload = done;
    after.src = set.after;
    before.src = set.before;
  });
}

// ---- zoom (wheel, cursor-anchored) ----
stage.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const ns = Math.min(40, Math.max(0.05, view.scale * factor));
    const k = ns / view.scale;
    view.x = cx - (cx - view.x) * k;
    view.y = cy - (cy - view.y) * k;
    view.scale = ns;
    applyTransform();
  },
  { passive: false }
);

// ---- pan (drag on stage) vs swipe (drag on handle) ----
let mode = null;
let start = null;

function pointFromEvent(e) {
  return { x: e.clientX, y: e.clientY };
}

stage.addEventListener("pointerdown", (e) => {
  const onHandle = e.target === handle || e.target === divider;
  mode = onHandle ? "swipe" : "pan";
  start = { ...pointFromEvent(e), vx: view.x, vy: view.y, split };
  stage.setPointerCapture(e.pointerId);
  if (mode === "pan") stage.classList.add("panning");
});

stage.addEventListener("pointermove", (e) => {
  if (!mode) return;
  const p = pointFromEvent(e);
  if (mode === "pan") {
    view.x = start.vx + (p.x - start.x);
    view.y = start.vy + (p.y - start.y);
    applyTransform();
  } else {
    // convert screen dx to image-space fraction
    const dxImg = (p.x - start.x) / view.scale;
    split = Math.min(1, Math.max(0, start.split + dxImg / img.w));
    applySplit();
  }
});

stage.addEventListener("pointerup", (e) => {
  mode = null;
  stage.classList.remove("panning");
  stage.releasePointerCapture(e.pointerId);
});

// ---- controls ----
$("combo").addEventListener("change", (e) => loadSet(e.target.value));
$("fit").addEventListener("click", fit);
$("tgl-unc").addEventListener("change", (e) => {
  $("ov-unc").hidden = !e.target.checked;
});
$("tgl-gate").addEventListener("change", (e) => {
  $("ov-gate").hidden = !e.target.checked;
});
$("tgl-qa").addEventListener("change", (e) => {
  $("ov-qa").hidden = !e.target.checked;
});
window.addEventListener("resize", fit);

// ---- stats panel ----
function renderStats(meta) {
  const dl = $("stats");
  const rows = [];
  const r = meta.run || {};
  const c = r.consistency_overall || {};
  rows.push(["model", r.model || "—"]);
  if (r.train_step != null) rows.push(["train step", r.train_step]);
  rows.push(["output px", `${meta.width} × ${meta.height}`]);
  rows.push(["SR pixel", `${(meta.sr_res_m || [])[0] ?? "?"} m`]);
  rows.push(["CRS", meta.sr_crs || "—"]);
  if (r.confidence_mean_reflectance != null)
    rows.push(["confidence mean · p95",
      `${r.confidence_mean_reflectance.toExponential(2)} · ${(r.confidence_p95_reflectance ?? 0).toExponential(2)}`]);
  if (r.gate_mean != null)
    rows.push(["gate mean (1 = CNN)", r.gate_mean.toFixed(3)]);
  if (c.rmse_10m != null)
    rows.push(["degrade↔LR · 10 m", `RMSE ${c.rmse_10m} · SAM ${c.sam_10m}°`]);
  if (r.n_tiles != null)
    rows.push(["tiles (flagged)", `${r.n_tiles} (${r.n_tiles_flagged ?? r.n_failed ?? 0})`]);
  if (meta.qa_flagged_fraction != null)
    rows.push(["flagged area", (meta.qa_flagged_fraction * 100).toFixed(2) + "%"]);
  if (meta.confidence_scale_max != null)
    rows.push(["confidence p98 (scale)", meta.confidence_scale_max.toExponential(2)]);
  dl.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  const kind = meta.confidence_kind;
  $("note").textContent = kind === "tta_spread"
    ? "Confidence here is a test-time-augmentation spread — a proxy. MC-dropout confidence ships with the DRISHTI model."
    : kind === "mc_dropout_std"
    ? "Confidence is the per-pixel spread over MC-dropout passes: bright = the model was guessing. The gate map (blue) marks where the transformer branch carried the reconstruction."
    : "";
  if (r.model) $("tag-right").textContent = `SR · ${(r.resolution_m ?? 2.5)} m · ${String(r.model).split(" ")[0]}`;
}

// ---- boot ----
(async function main() {
  let meta = {};
  try {
    meta = await fetch("assets/preview.json").then((r) => r.json());
    renderStats(meta);
    if (meta.confidence_scale_max != null) $("ov-unc").src = "assets/confidence.png";
    if (meta.gate_mean != null) $("ov-gate").src = "assets/gate.png";
    if (meta.qa_flagged_fraction != null) $("ov-qa").src = "assets/qa_mask.png";
  } catch (e) {
    $("stats").innerHTML = "<dt>error</dt><dd>run make_preview.py first</dd>";
  }
  await loadSet("rgb");
  for (const ov of [$("ov-unc"), $("ov-gate"), $("ov-qa")]) ov.style.width = img.w + "px";
  fit();
})();
