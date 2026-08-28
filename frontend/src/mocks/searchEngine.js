/** Deterministic local mock keyframe database + search engine. */
import realA from "../assets/0112.webp";
import realB from "../assets/0175.webp";
import realC from "../assets/0232.webp";
import { toTimecode, esc } from "../shared/format.js";

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function makeThumb({ id, seed, timecode, camera, videoKey, frameKey, folderKey }) {
  const rnd = mulberry32(seed);
  const hue = 18 + Math.floor(rnd() * 44);
  const hue2 = 170 + Math.floor(rnd() * 50);
  const rec = rnd() > 0.35;
  const gid = String(id).replace(/[^A-Za-z0-9_-]/g, "");
  const bars = Array.from({ length: 7 }, () => {
    const x = 40 + rnd() * 360;
    const y = 80 + rnd() * 120;
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(8 + rnd() * 26).toFixed(1)}" height="${(4 + rnd() * 12).toFixed(1)}" fill="hsl(${hue2} 60% 55% / 0.25)" />`;
  }).join("");
  const dots = Array.from({ length: 40 }, () => {
    const x = rnd() * 480;
    const y = rnd() * 270;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(rnd() * 1.2).toFixed(2)}" fill="hsl(${hue} 40% 70% / 0.35)" />`;
  }).join("");
  const lines = Array.from({ length: 9 }, () =>
    `<line x1="0" y1="${(rnd() * 240).toFixed(1)}" x2="480" y2="${(rnd() * 240).toFixed(1)}" stroke="hsl(${hue2} 70% 60% / 0.10)" stroke-width="1" />`
  ).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="480" height="270" viewBox="0 0 480 270">
  <defs>
    <linearGradient id="bg${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="hsl(${hue} 45% 9%)" />
      <stop offset="1" stop-color="hsl(${hue} 55% 3%)" />
    </linearGradient>
    <pattern id="sc${gid}" width="4" height="4" patternUnits="userSpaceOnUse">
      <rect width="4" height="4" fill="hsl(${hue2} 60% 55% / 0.04)" />
      <rect width="4" height="1" fill="hsl(${hue2} 60% 70% / 0.05)" />
    </pattern>
    <radialGradient id="vg${gid}" cx="0.5" cy="0.5" r="0.75">
      <stop offset="0.55" stop-color="black" stop-opacity="0" />
      <stop offset="1" stop-color="black" stop-opacity="0.65" />
    </radialGradient>
  </defs>
  <rect width="480" height="270" fill="url(#bg${gid})" />
  <rect width="480" height="270" fill="url(#sc${gid})" />
  <g opacity="0.9">${lines}</g>
  <g opacity="0.85">${bars}</g>
  <rect x="60" y="60" width="360" height="150" rx="4" fill="none" stroke="hsl(${hue2} 60% 60% / 0.4)" stroke-width="1.5" transform="rotate(${(rnd() - 0.5) * 4} 240 135)" />
  ${rec ? `<circle cx="432" cy="24" r="5" fill="#ff3b30"><animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite" /></circle>` : ""}
  ${dots}
  <rect width="480" height="270" fill="url(#vg${gid})" />
  <text x="14" y="26" font-family="monospace" font-size="13" fill="#ffd166" letter-spacing="1">${rec ? "-- REC  " : "▸ PLAY  "}${esc(timecode)}</text>
  <text x="14" y="44" font-family="monospace" font-size="11" fill="#7dd3fc" letter-spacing="1">${esc(camera)} • AK4K • ${gid}</text>
  <text x="466" y="258" text-anchor="end" font-family="monospace" font-size="11" fill="#94a3b8">${esc(folderKey)} / ${esc(videoKey)} / ${esc(frameKey)}</text>
  <text x="466" y="240" text-anchor="end" font-family="monospace" font-size="11" fill="#94a3b8">2026-08-02 AK-${Math.floor(rnd() * 900) + 100}</text>
  <line x1="0" y1="258" x2="480" y2="258" stroke="hsl(${hue2} 60% 60% / 0.35)" stroke-width="2" />
</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

const VIDEOS = [
  { folderKey: "ARC_DOCK", videoKey: "cam01_boarding", camera: "CAM 01", seedBase: 110 },
  { folderKey: "TRAFFIC_CTRL", videoKey: "cam04_overpass", camera: "CAM 04", seedBase: 220 },
  { folderKey: "METRO_LINE_A", videoKey: "platform_west", camera: "CAM 07", seedBase: 330 },
  { folderKey: "AIRPORT_TERM", videoKey: "gate_b12_baggage", camera: "CAM 12", seedBase: 440 },
  { folderKey: "SITE_DELIVERY", videoKey: "north_entrance", camera: "CAM 19", seedBase: 550 },
];
const FRAMES_PER_VIDEO = 16;
const OCR_TOKENS = ["NORTH GATE", "PLATFORM 2", "BAGGAGE A", "EXIT 4B", "DELIVERY BAY 7", "CHECK-IN ROW C", "ESCALATOR 1", "SECURITY SCAN", "PASSENGER LOUNGE", "PEDESTRIAN XING", "DOCK 12", "OVERHEAD 40"];
const OD_CLASSES = ["person", "vehicle", "truck", "suitcase", "bicycle", "bus", "forklift", "guard", "traffic_light", "car", "crowd", "luggage_cart"];
const QUESTION_BANK = [
  "A person in a high-visibility vest is walking beside the loading dock.",
  "Two delivery trucks are parked in parallel at the north entrance.",
  "The traffic signal cycles to green and the queue begins to move.",
  "A passenger with a red suitcase waits near gate column B12.",
  "The metro platform indicator shows the next train arriving in 3 minutes.",
  "A forklift crosses the delivery bay between two containers.",
  "An escalator is empty and running during the low-traffic window.",
];

function pickOcr(seed) {
  const r = mulberry32(seed);
  const out = [];
  const n = 1 + Math.floor(r() * 3);
  for (let i = 0; i < n; i++) out.push(OCR_TOKENS[Math.floor(r() * OCR_TOKENS.length)]);
  return out.join(" | ");
}

function pickOd(seed) {
  const r = mulberry32(seed);
  const out = [];
  const n = 1 + Math.floor(r() * 3);
  for (let i = 0; i < n; i++) out.push(OD_CLASSES[Math.floor(r() * OD_CLASSES.length)]);
  return out.filter((v, i, a) => a.indexOf(v) === i);
}

let poolCache = null;
export function getFramePool() {
  if (poolCache) return poolCache;
  poolCache = [];
  VIDEOS.forEach((v, vi) => {
    for (let fi = 0; fi < FRAMES_PER_VIDEO; fi++) {
      const timestamp = +(fi * 8.7 + vi * 1.3).toFixed(2);
      const frameKey = String(100001 + vi * 1000 + fi * 7).padStart(6, "0");
      const id = `FR-${v.camera.replace(" ", "")}-${String(fi + 1).padStart(3, "0")}`;
      const seed = hashString(`${v.seedBase}:${fi}`);
      const fps = 25;
      poolCache.push({
        id,
        gid: vi * FRAMES_PER_VIDEO + fi + 1,
        folderKey: v.folderKey,
        videoKey: v.videoKey,
        camera: v.camera,
        frameKey,
        frameName: `${v.videoKey}_${frameKey}`,
        globalFrameId: vi * FRAMES_PER_VIDEO + fi + 1,
        timestamp,
        timecode: toTimecode(timestamp, fps),
        fps,
        width: 1920,
        height: 1080,
        seed,
        faissIndex: vi * 1000 + fi,
        motion: +(0.1 + mulberry32(seed)() * 0.8).toFixed(2),
        ocrText: pickOcr(seed),
        odClasses: pickOd(seed),
        link: `/archive/${v.folderKey}/${v.videoKey}.mp4`,
        real: false,
        image: makeThumb({ id, seed, timecode: toTimecode(timestamp, fps), camera: v.camera, videoKey: v.videoKey, frameKey, folderKey: v.folderKey }),
      });
    }
  });
  return poolCache;
}

const realKeyframes = [
  { url: realA, id: "LIVE-FEED-A", folderKey: "REAL_FEED", videoKey: "broadcast_feed", camera: "CAM 09", seed: 9001, faissIndex: 9001 },
  { url: realB, id: "LIVE-FEED-B", folderKey: "REAL_FEED", videoKey: "broadcast_feed", camera: "CAM 10", seed: 9002, faissIndex: 9002 },
  { url: realC, id: "LIVE-FEED-C", folderKey: "REAL_FEED", videoKey: "broadcast_feed", camera: "CAM 11", seed: 9003, faissIndex: 9003 },
];

/** Mock search: ranks the frame pool for the active query/pivot. */
export function mockSearch(tab, pivot = null) {
  return new Promise((resolve) => {
    const pool = getFramePool();
    const topk = Math.min(Math.max(Number(tab.params?.topk) || 24, 1), pool.length + realKeyframes.length);
    const q = (tab.query || "").trim().toLowerCase();
    const type = tab.searchType || "TEXT";

    let scored = pool.map((f) => ({ f, raw: 0 }));

    if (type === "IMAGE" && pivot) {
      const base = typeof pivot.faissIndex === "number" ? pivot.faissIndex : pivot.seed || 0;
      scored.forEach((s) => {
        const d = Math.abs(s.f.seed - base) + Math.abs(s.f.faissIndex - base) * 0.02;
        s.raw = Math.max(0, 1 - d / 900);
      });
    } else if (type === "OCR+OD") {
      const tokens = q.split(/\s+/).filter(Boolean);
      scored.forEach((s) => {
        let hit = 0;
        tokens.forEach((t) => {
          if (s.f.ocrText.toLowerCase().includes(t)) hit += 2;
          if (s.f.odClasses.some((c) => c.toLowerCase().includes(t))) hit += 1.5;
          if (s.f.folderKey.toLowerCase().includes(t) || s.f.videoKey.toLowerCase().includes(t)) hit += 1;
        });
        s.raw = hit > 0 ? 0.5 + Math.min(hit, 4) * 0.12 : (hashString(q + s.f.id) / 4294967296) * 0.12;
      });
    } else if (type === "TEMPORAL") {
      scored.forEach((s) => {
        s.raw = (hashString(q + s.f.id) / 4294967296) * 0.6 + s.f.motion * 0.4;
      });
    } else {
      scored.forEach((s) => {
        let raw = hashString(q + s.f.id) / 4294967296;
        if (
          q &&
          (s.f.folderKey.toLowerCase().includes(q) ||
            s.f.videoKey.toLowerCase().includes(q) ||
            s.f.camera.toLowerCase().replace(" ", "").includes(q.replace(" ", "")))
        ) {
          raw += 0.5;
        }
        s.raw = raw;
      });
    }

    if (type === "TEXT" || type === "QA") {
      realKeyframes.forEach((r, i) => {
        scored.push({
          f: {
            id: r.id,
            gid: 9999 + i,
            folderKey: r.folderKey,
            videoKey: r.videoKey,
            camera: r.camera,
            frameKey: r.id.split("-").pop(),
            frameName: r.id,
            globalFrameId: 90000 + i,
            timestamp: +(60 + i * 22.3).toFixed(1),
            timecode: toTimecode(60 + i * 22.3),
            fps: 25,
            width: 1920,
            height: 1080,
            seed: r.seed,
            faissIndex: r.faissIndex,
            motion: 0.2,
            ocrText: "live feed timestamp overlay",
            odClasses: ["broadcast"],
            link: `/archive/${r.folderKey}/${r.videoKey}.mp4`,
            real: true,
            image: r.url,
          },
          raw: (hashString(q + r.id) / 4294967296) * 0.6 + 0.35,
        });
      });
    }

    scored.sort((a, b) => b.raw - a.raw);
    const items = scored.slice(0, topk).map((s, i) => ({
      ...s.f,
      score: +(Math.min(1, 0.4 + s.raw * 0.6)).toFixed(3),
      rank: i + 1,
      ...(type === "QA" ? { answer: QUESTION_BANK[hashString(q + s.f.id) % QUESTION_BANK.length] } : {}),
    }));

    const latency = 140 + (hashString(q + type) % 380);
    setTimeout(() => {
      resolve({ items, totalItems: items.length, latency, type, mode: "LOCAL MOCK ENGINE" });
    }, latency);
  });
}
