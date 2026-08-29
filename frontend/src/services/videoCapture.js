/**
 * Client for the backend video playback / frame-capture endpoints.
 *
 *   GET  /users/videos/{video_id}/playback?frame_idx=<optional>
 *   POST /users/videos/{video_id}/capture   { playback_time_seconds }
 *
 * The backend registers these under both `/users` and the root prefix, so the
 * URL builder here works whether or not `VITE_SEARCH_API_BASE_URL` already
 * ends in `/users`, and falls back to a same-origin relative path.
 */

import { getSearchConfig } from "./backendSearch.js";
import { toTimecode } from "../shared/format.js";

export class VideoCaptureError extends Error {
  constructor(message, { status, cause } = {}) {
    super(message, { cause });
    this.name = "VideoCaptureError";
    this.status = status;
  }
}

/** Build `<base>/users/videos/...`, tolerating a base with/without `/users`. */
export function videoApiUrl(baseUrl, path) {
  const cleanPath = String(path || "").replace(/^\/+/, "");
  const trimmed = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!trimmed) return `/users/${cleanPath}`;
  const usersBase = /\/users$/i.test(trimmed) ? trimmed : `${trimmed}/users`;
  return `${usersBase}/${cleanPath}`;
}

function firstFinite(...values) {
  for (const value of values) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return 0;
}

/** Stable id shared by the UI dedupe and the export dedupe. */
export function captureCandidateId(videoId, frameIdx) {
  return `capture:${String(videoId)}:${Number(frameIdx)}`;
}

/**
 * Resolve the backend `preview_url` (a relative captured-frame path, or an
 * absolute/data URL) into something the browser can load directly.
 */
export function normalizePreviewUrl(raw, baseUrl) {
  const value = String(raw || "").trim();
  if (!value) return "";
  if (/^(https?:|data:|blob:)/i.test(value)) return value;
  return videoApiUrl(baseUrl, value);
}

/** Normalize the backend playback envelope into a flat camelCase record. */
export function normalizePlaybackItem(raw) {
  const item = raw && typeof raw === "object" ? raw : {};
  return {
    videoId: String(item.video_id || ""),
    watchUrl: String(item.watch_url || ""),
    fps: firstFinite(item.fps),
    durationSeconds: firstFinite(item.duration_seconds),
    playbackOffsetSeconds: firstFinite(item.playback_offset_seconds),
    frameIdx: item.frame_idx == null ? null : Number(item.frame_idx),
    startSeconds: item.start_seconds == null ? null : Number(item.start_seconds),
  };
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

/**
 * Fetch playback metadata for a video, optionally resolving the player start
 * time for a specific dataset frame index.
 */
export async function fetchPlayback(
  videoId,
  frameIdx = null,
  { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {},
) {
  const id = String(videoId || "").trim();
  if (!id) throw new VideoCaptureError("A video id is required.");
  if (typeof fetchImpl !== "function") {
    throw new VideoCaptureError("Fetch is unavailable in this environment.");
  }

  const query = frameIdx == null || frameIdx === "" ? "" : `?frame_idx=${encodeURIComponent(frameIdx)}`;
  const url = videoApiUrl(config.baseUrl, `videos/${encodeURIComponent(id)}/playback${query}`);

  let response;
  try {
    response = await fetchImpl(url);
  } catch (cause) {
    throw new VideoCaptureError("The playback service is unavailable.", { cause });
  }
  const payload = await readJson(response);
  if (!response.ok || payload?.success === false) {
    throw new VideoCaptureError(
      payload?.message || payload?.detail || `Playback lookup failed (HTTP ${response.status}).`,
      { status: response.status },
    );
  }
  const first = payload?.data?.items?.[0];
  if (!first) throw new VideoCaptureError("Playback response was empty.");
  return normalizePlaybackItem(first);
}

/** Convert a player timestamp into a 0-based dataset frame index. */
export async function captureFrame(
  videoId,
  playbackTimeSeconds,
  { config = getSearchConfig(), fetchImpl = globalThis.fetch } = {},
) {
  const id = String(videoId || "").trim();
  if (!id) throw new VideoCaptureError("A video id is required.");
  const seconds = Number(playbackTimeSeconds);
  if (!Number.isFinite(seconds)) {
    throw new VideoCaptureError("A finite playback time is required.");
  }
  if (typeof fetchImpl !== "function") {
    throw new VideoCaptureError("Fetch is unavailable in this environment.");
  }

  const url = videoApiUrl(config.baseUrl, `videos/${encodeURIComponent(id)}/capture`);
  let response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playback_time_seconds: seconds }),
    });
  } catch (cause) {
    throw new VideoCaptureError("The capture service is unavailable.", { cause });
  }
  const payload = await readJson(response);
  if (!response.ok || payload?.success === false) {
    throw new VideoCaptureError(
      payload?.message || payload?.detail || `Capture failed (HTTP ${response.status}).`,
      { status: response.status },
    );
  }
  const data = payload?.data || {};
  return {
    videoId: String(data.video_id || id),
    playbackTimeSeconds: firstFinite(data.playback_time_seconds, seconds),
    sourceTimeSeconds: firstFinite(data.source_time_seconds, seconds),
    fps: firstFinite(data.fps),
    frameIdx: Number(data.frame_idx),
    // Exact server-extracted still for this frame, or "" when unavailable.
    previewUrl: normalizePreviewUrl(data.preview_url, config.baseUrl),
    previewError: data.preview_error ? String(data.preview_error) : null,
  };
}

/**
 * Turn a capture result into a Selection Tray candidate that matches the
 * workstation card shape used by the export pipeline. The preview image is the
 * exact server-extracted still for the submitted frame (`capture.previewUrl`);
 * there is no fallback to the reviewed frame's thumbnail. When extraction is
 * unavailable the candidate stays selectable/exportable with an empty `image`.
 */
export function buildCaptureCandidate(reviewItem, capture) {
  const videoId = String(capture?.videoId || reviewItem?.videoKey || "");
  const frameIdx = Number(capture?.frameIdx);
  const fps = firstFinite(capture?.fps, reviewItem?.fps, 25);
  const sourceTime = firstFinite(capture?.sourceTimeSeconds);
  const previewUrl = String(capture?.previewUrl || "");
  const previewError = capture?.previewError ? String(capture.previewError) : null;
  const folderKey = String(
    reviewItem?.folderKey || reviewItem?.backend?.folder_key || videoId.split("_")[0] || "UNKNOWN",
  );

  return {
    id: captureCandidateId(videoId, frameIdx),
    gid: frameIdx,
    globalFrameId: frameIdx,
    submissionFrameId: frameIdx,
    folderKey,
    videoKey: videoId,
    camera: "CAPTURE",
    frameKey: String(frameIdx),
    frameName: `${videoId}_${frameIdx}`,
    timestamp: sourceTime,
    timecode: toTimecode(sourceTime, fps || 25),
    fps: fps || 25,
    image: previewUrl,
    previewUrl,
    previewError,
    hasPreview: Boolean(previewUrl),
    link: reviewItem?.link || reviewItem?.backend?.watch_url || "",
    real: true,
    rank: null,
    score: null,
    captured: true,
    capturePlaybackSeconds: firstFinite(capture?.playbackTimeSeconds),
    answer: reviewItem?.answer,
    backend: {
      video_id: videoId,
      frame_idx: frameIdx,
      submission_frame_id: frameIdx,
      source_time_seconds: sourceTime,
      playback_time_seconds: firstFinite(capture?.playbackTimeSeconds),
      fps: fps || 25,
      captured: true,
      preview_url: previewUrl || null,
    },
  };
}
