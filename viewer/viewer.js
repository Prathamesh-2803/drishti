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
$("tgl-qa").addEventListener("change", (e) => {
  $("ov-qa").hidden = !e.target.checked;
});
window.addEventListener("resize", fit);

// ---- stats panel ----
function renderStats(meta) {
  const dl = $("stats");
  const rows = [];
  const q = meta.qa || {};
  const c = q.consistency_overall || {};
  rows.push(["model", q.model || "—"]);
  rows.push(["output px", `${meta.width} × ${meta.height}`]);
  rows.push(["SR pixel", `${(meta.sr_res_m || [])[0] ?? "?"} m`]);
  rows.push(["CRS", meta.sr_crs || "—"]);
  if (c.rmse_10m != null)
    rows.push(["degrade↔LR · 10 m", `RMSE ${c.rmse_10m} · SAM ${c.sam_10m}°`]);
  if (c.rmse_20m != null)
    rows.push(["degrade↔LR · 20 m", `RMSE ${c.rmse_20m} · SAM ${c.sam_20m}°`]);
  if (q.n_tiles != null)
    rows.push(["tiles (failed)", `${q.n_tiles} (${q.n_failed})`]);
  if (meta.qa_failed_fraction != null)
    rows.push(["fallback area", (meta.qa_failed_fraction * 100).toFixed(2) + "%"]);
  if (meta.uncertainty_scale_max != null)
    rows.push(["uncertainty p98", meta.uncertainty_scale_max.toExponential(2)]);
  dl.innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join("");
}

// ---- boot ----
(async function main() {
  try {
    const meta = await fetch("assets/preview.json").then((r) => r.json());
    renderStats(meta);
    if (meta.uncertainty_scale_max != null) $("ov-unc").src = "assets/uncertainty.png";
    if (meta.qa_failed_fraction != null) $("ov-qa").src = "assets/qa_mask.png";
  } catch (e) {
    $("stats").innerHTML = "<dt>error</dt><dd>run make_preview.py first</dd>";
  }
  await loadSet("rgb");
  // overlays share the SR grid
  for (const ov of [$("ov-unc"), $("ov-qa")]) ov.style.width = img.w + "px";
  fit();
})();
