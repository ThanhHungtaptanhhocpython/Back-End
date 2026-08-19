/** Frame time formatting helpers shared across features. */

export function toTimecode(sec, fps = 25) {
  const pad = (n) => String(n).padStart(2, "0");
  const fr = Math.floor((sec % 1) * fps) % fps;
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  return `${pad(h)}:${pad(m)}:${pad(s)}:${pad(fr)}`;
}

export function fmtDur(sec) {
  const pad = (n) => String(n).padStart(2, "0");
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
