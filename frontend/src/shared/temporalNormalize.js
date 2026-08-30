/**
 * TRAKE (temporal) response normalizer.
 *
 * A temporal search returns *sequences* (one keyframe per requested event), not
 * a flat frame list. This keeps the nested shape instead of flattening it into
 * result cards, and pins each frame's submission id to the original per-video
 * frame index — never a FAISS / vector id.
 */

import { toTimecode } from "./format.js";

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function finiteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function imageSource(value) {
  if (!value || typeof value !== "string") return "";
  if (/^(data:|blob:|https?:\/\/|\/)/i.test(value)) return value;
  return `data:image/webp;base64,${value}`;
}

/** Integer submission frame id — original per-video frame index only. */
export function resolveSubmissionFrameId(raw) {
  const direct = firstDefined(
    raw?.submission_frame_id,
    raw?.submissionFrameId,
    raw?.frame_index,
    raw?.frame_idx,
  );
  if (direct !== undefined) {
    const parsed = Number.parseInt(String(direct).replace(/[^\d-]/g, ""), 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  // Keyframe files are named `<frame_idx>.webp`; the last digit run in a label
  // like `L21_V001_000660` / `000660.webp` is the per-video frame index.
  const label = String(
    firstDefined(raw?.frame_key, raw?.frameKey, raw?.frame_id, raw?.frame_name, raw?.frameName, ""),
  ).trim();
  const digits = label.match(/(\d+)(?=\D*$)/);
  if (digits) {
    const parsed = Number.parseInt(digits[1], 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function normalizeFrame(raw, index, sequence, videoKey) {
  const record = raw && typeof raw === "object" ? raw : {};
  const eventIndex = finiteNumber(firstDefined(record.event_index, record.eventIndex, index + 1), index + 1);
  const timestamp = finiteNumber(firstDefined(record.timestamp, sequence?.timestamps?.[index]), 0);
  const fps = 25;
  const frameKey = String(firstDefined(record.frame_key, record.frameKey, record.frame_id, ""));
  const submissionFrameId = resolveSubmissionFrameId(record);
  const frameVideoKey = String(firstDefined(record.video_key, record.videoKey, videoKey, "unknown-video"));
  const evidenceRow = sequence?.evidence?.[index] || record.evidence || {};

  return {
    id: `${sequence?.id ?? "seq"}:e${eventIndex}`,
    eventIndex,
    eventQuery: String(firstDefined(record.event_query, sequence?.event_queries?.[index], "")),
    eventQueryEn: String(firstDefined(record.event_query_en, sequence?.event_queries_en?.[index], "")),
    videoKey: frameVideoKey,
    folderKey: String(firstDefined(record.folder_key, record.folderKey, frameVideoKey.split("_")[0], "UNKNOWN")),
    camera: "TRAKE",
    frameKey,
    frameName: String(firstDefined(record.frame_name, record.frameName, frameKey || `${frameVideoKey}_${submissionFrameId ?? eventIndex}`)),
    submissionFrameId,
    globalFrameId: submissionFrameId,
    timestamp,
    timecode: toTimecode(timestamp, fps),
    fps,
    image: imageSource(firstDefined(record.image, record.thumbnail, record.image_url)),
    link: String(firstDefined(record.link, record.youtube_url, "") ?? ""),
    evidence: {
      scores: evidenceRow.scores || evidenceRow.evidence_scores || {},
      text: evidenceRow.text || evidenceRow.evidence_text || {},
    },
    score: finiteNumber(firstDefined(record.score, sequence?.score), 0),
    real: true,
    unresolved: submissionFrameId === null,
  };
}

function normalizeSequence(raw, index) {
  const record = raw && typeof raw === "object" ? raw : {};
  const videoKey = String(firstDefined(record.video_id, record.videoKey, record.video_key, "unknown-video"));
  const rawFrames = Array.isArray(record.frames) ? record.frames : [];
  const id = String(
    firstDefined(
      record.sequence_id,
      record.sequenceId,
      videoKey !== "unknown-video" && rawFrames.length
        ? `${videoKey}#${rawFrames.map((f) => resolveSubmissionFrameId(f) ?? "?").join("-")}`
        : undefined,
      `seq-${index + 1}`,
    ),
  );

  const seqShell = {
    id,
    event_queries: record.event_queries,
    event_queries_en: record.event_queries_en,
    evidence: record.evidence,
    timestamps: record.timestamps,
    score: finiteNumber(firstDefined(record.verification_score, record.score), 0),
  };
  const frames = rawFrames.map((frame, frameIndex) => normalizeFrame(frame, frameIndex, seqShell, videoKey));

  const timestamps = frames.map((frame) => frame.timestamp);
  const orderOk = timestamps.every((value, i) => i === 0 || value >= timestamps[i - 1]);
  const sameVideo = frames.every((frame) => frame.videoKey === videoKey);
  const resolved = frames.every((frame) => !frame.unresolved);

  return {
    id,
    videoKey,
    score: seqShell.score,
    baseScore: finiteNumber(record.base_score, 0),
    timestamps,
    temporalGaps: Array.isArray(record.temporal_gaps) ? record.temporal_gaps : [],
    verification: {
      method: String(firstDefined(record.verification?.method, "temporal_evidence")),
      status: String(firstDefined(record.verification?.summary?.status, record.verification?.status, "")),
      vlmDecision: record.vlm_decision ?? null,
      vlmScore: record.vlm_score ?? null,
      vlmReason: record.vlm_reason ?? "",
      matchedEvents: Array.isArray(record.vlm_matched_events) ? record.vlm_matched_events : [],
      missingEvents: Array.isArray(record.vlm_missing_events) ? record.vlm_missing_events : [],
    },
    valid: orderOk && sameVideo && resolved && frames.length > 0,
    orderOk,
    sameVideo,
    resolved,
    edited: false,
    chosen: false,
    rank: index + 1,
    frames,
  };
}

/** Normalize a FastAPI BaseResponse whose items are temporal sequences. */
export function normalizeTemporalResponse(payload, { latency = 0 } = {}) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Backend returned an invalid temporal response.");
  }
  if (payload.success !== true) {
    throw new Error(payload.message || "Temporal search was not successful.");
  }
  const items = payload?.data?.items;
  if (!Array.isArray(items)) {
    throw new Error("Temporal response did not include a sequence list.");
  }

  const sequences = items
    .map((item, index) => normalizeSequence(item, index))
    .map((sequence, index) => ({ ...sequence, rank: index + 1 }));

  return {
    sequences,
    totalItems: sequences.length,
    meta: payload?.data?.meta || null,
    latency,
    type: "TEMPORAL",
    mode: "FASTAPI LIVE",
    source: "live",
  };
}

/** Detect whether a payload's items are temporal sequences (nested `frames`). */
export function looksLikeTemporalPayload(payload) {
  const items = payload?.data?.items;
  return Array.isArray(items) && items.some((item) => Array.isArray(item?.frames));
}

/**
 * Re-order sequences after a manual edit or an explicit "use this" pick:
 * chosen / edited sequences first (rank 1..), then the remaining automatic
 * sequences, dropping any later sequence whose whole (video + ordered frame
 * ids) tuple duplicates an earlier one.
 */
export function reindexTemporalSequences(sequences) {
  const picked = [];
  const auto = [];
  for (const sequence of sequences || []) {
    (sequence?.chosen || sequence?.edited ? picked : auto).push(sequence);
  }
  const seen = new Set();
  const ordered = [];
  for (const sequence of [...picked, ...auto]) {
    const key = `${sequence.videoKey}|${(sequence.frames || []).map((f) => f.submissionFrameId ?? "?").join("-")}`;
    if (seen.has(key)) continue;
    seen.add(key);
    ordered.push(sequence);
  }
  return ordered.map((sequence, index) => ({ ...sequence, rank: index + 1 }));
}

/** One export row per sequence: `{ videoId, frameIds: number[] }` (nulls dropped upstream). */
export function temporalSequenceToExportRow(sequence) {
  return {
    videoId: sequence?.videoKey || "unknown-video",
    frameIds: (sequence?.frames || []).map((frame) =>
      frame?.submissionFrameId ?? resolveSubmissionFrameId(frame?.backend || frame),
    ),
  };
}
