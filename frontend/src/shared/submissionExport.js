function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function normalizeVideoName(value) {
  return String(value || "unknown-video").trim().replace(/\.(mp4|mov|avi|mkv|webm)$/i, "");
}

function videoNameForItem(item) {
  const rawVideo = item?.videoKey ?? item?.backend?.video_name ?? item?.backend?.video_id ?? item?.video_id ?? item?.videoName;
  const video = normalizeVideoName(rawVideo);
  const folder = String(item?.folderKey ?? item?.backend?.folder_key ?? item?.backend?.split ?? item?.split ?? "").trim();
  if (/^V\d+/i.test(video)) {
    if (/^L\d+/i.test(folder)) {
      return normalizeVideoName(`${folder}_${video}`);
    }
    const m = folder.match(/l(\d+)/i);
    if (m) {
      return `L${m[1]}_${video}`;
    }
  }
  return video;
}

function normalizeFrameId(item) {
  const raw = firstDefined(
    item?.submissionFrameId,
    item?.submission_frame_id,
    item?.backend?.submission_frame_id,
    item?.backend?.frame_idx,
    item?.frameKey,
    item?.frame_key,
    item?.backend?.frame_key,
    item?.frame_id,
    item?.backend?.frame_id,
    item?.globalFrameId,
    item?.global_frame_id,
    item?.backend?.global_frame_id,
    item?.id
  );
  const parsed = Number.parseInt(String(raw).replace(/[^\d-]/g, ""), 10);
  return Number.isFinite(parsed) ? String(parsed) : String(raw || "").trim();
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function sanitizeQueryFileName(name, queryType = "kis") {
  const cleaned = String(name || "").trim().replace(/[\\/:*?"<>|]+/g, "_");
  const base = cleaned || `query-1-${queryType.toLowerCase()}`;
  return base.toLowerCase().endsWith(".csv") ? base : `${base}.csv`;
}

export function queryTypeFromSearchType(searchType) {
  const type = String(searchType || "").toUpperCase();
  if (type === "QA" || type === "Q&A SEARCH") return "qa";
  if (type === "TEMPORAL" || type === "TEMPORAL SEARCH") return "trake";
  return "kis";
}

/** Drop later rows that repeat an earlier `video_id,frame_idx` pair. */
function dedupeByVideoFrame(items) {
  const seen = new Set();
  const kept = [];
  for (const item of items || []) {
    const key = `${videoNameForItem(item)}|${normalizeFrameId(item)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(item);
  }
  return kept;
}

function looksLikeSequenceList(items) {
  return Array.isArray(items) && items.some((item) => Array.isArray(item?.frames));
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Clamp a jittered row back to non-negative, strictly-increasing frame ids. */
function enforceOrder(ids) {
  const out = ids.map((value) => Math.max(0, Math.round(value)));
  for (let i = 1; i < out.length; i += 1) {
    if (out[i] <= out[i - 1]) out[i] = out[i - 1] + 1;
  }
  return out;
}

/**
 * Expand one chosen candidate's frame ids into up to `rows` ordered rows so the
 * submission blankets each event's short ground-truth interval `[sⱼ, eⱼ]`
 * (BTC rule: a submitted frame counts if it lands anywhere inside that window,
 * which is usually < 10 frames wide). Row 0 is always the exact chosen frames;
 * the rest are deterministic jitters within `±radius`.
 */
export function expandTemporalFrames(frameIds, { radius = 0, rows = 100 } = {}) {
  const base = (frameIds || []).map((value) => Number.parseInt(value, 10)).filter(Number.isFinite);
  const cap = Math.min(100, Math.max(1, Math.round(rows) || 1));
  if (base.length < 2 || radius <= 0) return [enforceOrder(base)];

  const seen = new Set();
  const out = [];
  const push = (ids) => {
    if (out.length >= cap) return;
    const ordered = enforceOrder(ids);
    const key = ordered.join(",");
    if (seen.has(key)) return;
    seen.add(key);
    out.push(ordered);
  };

  push(base); // row 0: exact chosen frames
  for (let d = 1; d <= radius; d += 1) {
    push(base.map((value) => value - d)); // coordinated slide back
    push(base.map((value) => value + d)); // coordinated slide forward
  }
  for (let j = 0; j < base.length; j += 1) {
    for (let d = -radius; d <= radius; d += 1) {
      if (d !== 0) {
        const ids = base.slice();
        ids[j] = base[j] + d;
        push(ids);
      }
    }
  }
  const rand = mulberry32((base[0] * 73856093) ^ (base.length * 19349663) ^ (radius * 83492791));
  for (let guard = 0; out.length < cap && guard < cap * 50; guard += 1) {
    push(base.map((value) => value + Math.round((rand() * 2 - 1) * radius)));
  }
  return out;
}

/**
 * TRAKE CSV: up to 100 rows, one full candidate per row:
 *   `video_id,frame_event_1,...,frame_event_N`
 * Manually-edited sequences are emitted first; a later row that repeats an
 * earlier row's whole (video + ordered frames) tuple is dropped. Selection Tray
 * frames are never folded in here.
 *
 * `options.jitterRadius > 0` expands the first (chosen) sequence into jittered
 * rows via `expandTemporalFrames`; the remaining sequences then fill up to 100.
 */
export function buildTemporalSubmissionCsv(sequences, options = {}) {
  const jitterRadius = Math.max(0, Math.round(Number(options.jitterRadius) || 0));
  const jitterRows = Math.min(100, Math.max(1, Math.round(Number(options.jitterRows) || 100)));

  const edited = [];
  const auto = [];
  for (const sequence of Array.isArray(sequences) ? sequences : []) {
    const frames = Array.isArray(sequence?.frames) ? sequence.frames : [];
    if (frames.length < 2) continue;
    const videoId = videoNameForItem({
      videoKey: sequence.videoKey ?? sequence.video_id ?? sequence.videoId,
      folderKey: sequence.folderKey ?? frames[0]?.folderKey,
    });
    const frameIds = frames.map((frame) => {
      const value = frame?.submissionFrameId ?? frame?.submission_frame_id;
      return Number.isFinite(Number(value)) ? Number.parseInt(value, 10) : null;
    });
    if (frameIds.some((value) => value === null)) continue;
    (sequence?.chosen || sequence?.edited ? edited : auto).push({ videoId, frameIds });
  }

  const ordered = [...edited, ...auto];
  const seen = new Set();
  const lines = [];
  const emit = (videoId, ids) => {
    if (lines.length >= 100) return;
    const line = [videoId, ...ids].join(",");
    if (seen.has(line)) return;
    seen.add(line);
    lines.push(line);
  };

  if (jitterRadius > 0 && ordered.length) {
    const [chosen, ...rest] = ordered;
    for (const ids of expandTemporalFrames(chosen.frameIds, { radius: jitterRadius, rows: jitterRows })) {
      emit(chosen.videoId, ids);
    }
    for (const row of rest) emit(row.videoId, row.frameIds);
  } else {
    for (const row of ordered) emit(row.videoId, row.frameIds);
  }
  return lines.join("\n");
}

export function buildSubmissionCsv(items, queryType, answer = "") {
  const type = String(queryType || "kis").toLowerCase();

  if (type === "trake") {
    if (looksLikeSequenceList(items)) {
      return buildTemporalSubmissionCsv(items);
    }
    // Legacy single-sequence shape: one line, ordered frames, no dedupe.
    const rows = (items || []).slice(0, 100);
    if (rows.length === 0) return "";
    const firstVideo = videoNameForItem(rows[0]);
    return [firstVideo, ...rows.map(normalizeFrameId)].join(",");
  }

  // KIS / QA: a captured frame and a search hit for the same
  // (video_id, frame_idx) must not produce two rows.
  const rows = dedupeByVideoFrame(items || []).slice(0, 100);

  if (type === "qa") {
    return rows
      .map((item) => [
        videoNameForItem(item),
        normalizeFrameId(item),
        csvCell(String(firstDefined(item?.answer, item?.backend?.answer, answer)).slice(0, 100)),
      ].join(","))
      .join("\n");
  }

  return rows
    .map((item) => [
      videoNameForItem(item),
      normalizeFrameId(item),
    ].join(","))
    .join("\n");
}

function crc32(text) {
  const bytes = new TextEncoder().encode(text);
  let crc = -1;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ -1) >>> 0;
}

function dosDateTime(date = new Date()) {
  const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
  const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { dosDate, dosTime };
}

function pushU16(out, value) {
  out.push(value & 0xff, (value >>> 8) & 0xff);
}

function pushU32(out, value) {
  out.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff);
}

function pushBytes(out, bytes) {
  for (const byte of bytes) out.push(byte);
}

function normalizeZipFiles(files) {
  const normalized = (files || [])
    .map((file) => ({
      name: sanitizeQueryFileName(file?.name, file?.queryType),
      content: String(file?.content ?? ""),
    }))
    .filter((file) => file.name);

  return normalized.length > 0 ? normalized : [{ name: "query-1-kis.csv", content: "" }];
}

export function makeSubmissionZip(files) {
  const encoder = new TextEncoder();
  const { dosDate, dosTime } = dosDateTime();
  const local = [];
  const central = [];
  const zipFiles = [{ name: "submission/", content: "" }, ...normalizeZipFiles(files)];

  zipFiles.forEach((file) => {
    // Match official sample archives: place CSV files inside submission/.
    const path = file.name.startsWith("submission/") ? file.name : `submission/${file.name}`;
    const nameBytes = encoder.encode(path);
    const contentBytes = encoder.encode(file.content);
    const checksum = crc32(file.content);
    const localOffset = local.length;

    pushU32(local, 0x04034b50);
    pushU16(local, 20);
    pushU16(local, 0x0800);
    pushU16(local, 0);
    pushU16(local, dosTime);
    pushU16(local, dosDate);
    pushU32(local, checksum);
    pushU32(local, contentBytes.length);
    pushU32(local, contentBytes.length);
    pushU16(local, nameBytes.length);
    pushU16(local, 0);
    pushBytes(local, nameBytes);
    pushBytes(local, contentBytes);

    pushU32(central, 0x02014b50);
    pushU16(central, 20);
    pushU16(central, 20);
    pushU16(central, 0x0800);
    pushU16(central, 0);
    pushU16(central, dosTime);
    pushU16(central, dosDate);
    pushU32(central, checksum);
    pushU32(central, contentBytes.length);
    pushU32(central, contentBytes.length);
    pushU16(central, nameBytes.length);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU16(central, 0);
    pushU32(central, 0);
    pushU32(central, localOffset);
    pushBytes(central, nameBytes);
  });

  const end = [];
  pushU32(end, 0x06054b50);
  pushU16(end, 0);
  pushU16(end, 0);
  pushU16(end, zipFiles.length);
  pushU16(end, zipFiles.length);
  pushU32(end, central.length);
  pushU32(end, local.length);
  pushU16(end, 0);

  return new Blob([new Uint8Array(local), new Uint8Array(central), new Uint8Array(end)], {
    type: "application/zip",
  });
}

export function makeSingleFileSubmissionZip(csvFileName, csvContent, queryType = "kis") {
  return makeSubmissionZip([{ name: csvFileName, content: csvContent, queryType }]);
}
