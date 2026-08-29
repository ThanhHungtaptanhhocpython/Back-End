import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCaptureCandidate,
  captureCandidateId,
  captureFrame,
  fetchPlayback,
  normalizePlaybackItem,
  normalizePreviewUrl,
  VideoCaptureError,
  videoApiUrl,
} from "../src/services/videoCapture.js";
import { buildVideoPlayback, youtubeVideoId } from "../src/services/videoPlayback.js";
import { buildSubmissionCsv } from "../src/shared/submissionExport.js";

test("videoApiUrl works with, without, and missing the /users suffix", () => {
  assert.equal(
    videoApiUrl("http://localhost:3000", "videos/L21_V001/playback"),
    "http://localhost:3000/users/videos/L21_V001/playback",
  );
  assert.equal(
    videoApiUrl("http://localhost:3000/users/", "videos/L21_V001/capture"),
    "http://localhost:3000/users/videos/L21_V001/capture",
  );
  assert.equal(
    videoApiUrl("", "videos/L21_V001/playback"),
    "/users/videos/L21_V001/playback",
  );
});

test("normalizePlaybackItem flattens the backend snake_case envelope", () => {
  const item = normalizePlaybackItem({
    video_id: "L21_V001",
    watch_url: "https://youtube.com/watch?v=abc123",
    fps: 29.97,
    duration_seconds: 1262,
    playback_offset_seconds: -172,
    frame_idx: 351,
    start_seconds: 11.7,
  });
  assert.deepEqual(item, {
    videoId: "L21_V001",
    watchUrl: "https://youtube.com/watch?v=abc123",
    fps: 29.97,
    durationSeconds: 1262,
    playbackOffsetSeconds: -172,
    frameIdx: 351,
    startSeconds: 11.7,
  });
});

test("fetchPlayback GETs the playback endpoint and returns the first item", async () => {
  let calledUrl;
  const meta = await fetchPlayback("L21_V001", 351, {
    config: { baseUrl: "http://localhost:3000/users", mode: "live" },
    fetchImpl: async (url) => {
      calledUrl = url;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          data: {
            total_items: 1,
            items: [
              {
                video_id: "L21_V001",
                watch_url: "https://youtube.com/watch?v=abc123",
                fps: 30,
                duration_seconds: 1262,
                playback_offset_seconds: 0,
                frame_idx: 351,
                start_seconds: 11.7,
              },
            ],
          },
        }),
      };
    },
  });

  assert.equal(calledUrl, "http://localhost:3000/users/videos/L21_V001/playback?frame_idx=351");
  assert.equal(meta.fps, 30);
  assert.equal(meta.startSeconds, 11.7);
});

test("captureFrame POSTs the timestamp and parses the frame index", async () => {
  let call;
  const result = await captureFrame("L21_V001", 11.7333, {
    config: { baseUrl: "http://localhost:3000", mode: "live" },
    fetchImpl: async (url, init) => {
      call = { url, init };
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          data: {
            video_id: "L21_V001",
            playback_time_seconds: 11.7333,
            source_time_seconds: 11.7333,
            fps: 30,
            frame_idx: 351,
          },
        }),
      };
    },
  });

  assert.equal(call.url, "http://localhost:3000/users/videos/L21_V001/capture");
  assert.deepEqual(JSON.parse(call.init.body), { playback_time_seconds: 11.7333 });
  assert.equal(result.frameIdx, 351);
  assert.equal(result.fps, 30);
  // No preview fields in the response -> empty preview, no error.
  assert.equal(result.previewUrl, "");
  assert.equal(result.previewError, null);
});

test("normalizePreviewUrl resolves relative captured-frame paths and passes URLs through", () => {
  assert.equal(normalizePreviewUrl("", "http://localhost:3000"), "");
  assert.equal(
    normalizePreviewUrl("videos/captures/L21_V001/351.webp", "http://localhost:3000"),
    "http://localhost:3000/users/videos/captures/L21_V001/351.webp",
  );
  assert.equal(
    normalizePreviewUrl("https://cdn.example/x.webp", "http://localhost:3000"),
    "https://cdn.example/x.webp",
  );
});

test("captureFrame normalizes preview_url and surfaces preview_error", async () => {
  const withPreview = await captureFrame("L21_V001", 11.7333, {
    config: { baseUrl: "http://localhost:3000", mode: "live" },
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        success: true,
        data: {
          video_id: "L21_V001",
          playback_time_seconds: 11.7333,
          source_time_seconds: 11.7333,
          fps: 30,
          frame_idx: 351,
          preview_url: "videos/captures/L21_V001/351.webp",
          preview_error: null,
        },
      }),
    }),
  });
  assert.equal(
    withPreview.previewUrl,
    "http://localhost:3000/users/videos/captures/L21_V001/351.webp",
  );
  assert.equal(withPreview.previewError, null);

  const noPreview = await captureFrame("L21_V001", 11.7333, {
    config: { baseUrl: "http://localhost:3000", mode: "live" },
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        success: true,
        data: {
          video_id: "L21_V001",
          playback_time_seconds: 11.7333,
          source_time_seconds: 11.7333,
          fps: 30,
          frame_idx: 351,
          preview_url: null,
          preview_error: "FFmpeg binary is not available on this server.",
        },
      }),
    }),
  });
  assert.equal(noPreview.previewUrl, "");
  assert.equal(noPreview.previewError, "FFmpeg binary is not available on this server.");
});

test("captureFrame surfaces backend 400s as VideoCaptureError", async () => {
  await assert.rejects(
    captureFrame("L21_V001", -3, {
      config: { baseUrl: "http://localhost:3000", mode: "live" },
      fetchImpl: async () => ({
        ok: false,
        status: 400,
        json: async () => ({ success: false, message: "must not be negative" }),
      }),
    }),
    (error) => error instanceof VideoCaptureError && error.status === 400,
  );
});

test("buildCaptureCandidate produces a tray candidate with the canonical id", () => {
  const reviewItem = {
    videoKey: "L21_V001",
    folderKey: "L21",
    image: "data:image/webp;base64,REVIEWTHUMB",
    link: "https://youtube.com/watch?v=abc123",
  };
  const candidate = buildCaptureCandidate(reviewItem, {
    videoId: "L21_V001",
    playbackTimeSeconds: 11.7333,
    sourceTimeSeconds: 11.7333,
    fps: 30,
    frameIdx: 351,
    previewUrl: "http://localhost:3000/users/videos/captures/L21_V001/351.webp",
  });

  assert.equal(candidate.id, "capture:L21_V001:351");
  assert.equal(candidate.id, captureCandidateId("L21_V001", 351));
  assert.equal(candidate.videoKey, "L21_V001");
  assert.equal(candidate.frameName, "L21_V001_351");
  assert.equal(candidate.submissionFrameId, 351);
  assert.equal(candidate.backend.frame_idx, 351);
  assert.equal(candidate.captured, true);
});

test("buildCaptureCandidate uses preview_url and never the reviewed frame image", () => {
  const reviewItem = { videoKey: "L21_V001", image: "data:image/webp;base64,REVIEWTHUMB" };
  const previewUrl = "http://localhost:3000/users/videos/captures/L21_V001/351.webp";
  const candidate = buildCaptureCandidate(reviewItem, {
    videoId: "L21_V001",
    sourceTimeSeconds: 11.7333,
    fps: 30,
    frameIdx: 351,
    previewUrl,
  });

  assert.equal(candidate.image, previewUrl);
  assert.equal(candidate.previewUrl, previewUrl);
  assert.equal(candidate.hasPreview, true);
  assert.notEqual(candidate.image, reviewItem.image);
  assert.equal(candidate.backend.preview_url, previewUrl);
});

test("buildCaptureCandidate keeps an empty image when extraction is unavailable", () => {
  const reviewItem = { videoKey: "L21_V001", image: "data:image/webp;base64,REVIEWTHUMB" };
  const candidate = buildCaptureCandidate(reviewItem, {
    videoId: "L21_V001",
    sourceTimeSeconds: 11.7333,
    fps: 30,
    frameIdx: 351,
    previewUrl: "",
    previewError: "yt-dlp is not installed on this server.",
  });

  assert.equal(candidate.image, "");
  assert.equal(candidate.hasPreview, false);
  assert.equal(candidate.previewError, "yt-dlp is not installed on this server.");
  assert.equal(candidate.backend.preview_url, null);
  // Still a valid, exportable candidate.
  assert.equal(candidate.submissionFrameId, 351);
  assert.equal(candidate.captured, true);
});

test("duplicate captures collapse to a single KIS CSV row", () => {
  const first = buildCaptureCandidate(
    { videoKey: "L21_V001" },
    { videoId: "L21_V001", sourceTimeSeconds: 11.7, fps: 30, frameIdx: 351 },
  );
  const second = buildCaptureCandidate(
    { videoKey: "L21_V001" },
    { videoId: "L21_V001", sourceTimeSeconds: 11.7, fps: 30, frameIdx: 351 },
  );
  const csv = buildSubmissionCsv([first, second], "kis");
  assert.equal(csv, "L21_V001,351");
});

test("youtubeVideoId parses watch, youtu.be, embed and shorts URLs", () => {
  assert.equal(youtubeVideoId("https://youtube.com/watch?v=abc123"), "abc123");
  assert.equal(youtubeVideoId("https://youtu.be/abc123"), "abc123");
  assert.equal(youtubeVideoId("https://www.youtube.com/embed/abc123"), "abc123");
  assert.equal(youtubeVideoId("https://www.youtube.com/shorts/abc123"), "abc123");
  assert.equal(youtubeVideoId("not a url"), "");
});

test("buildVideoPlayback applies the supplied offset without any hardcoded table", () => {
  const item = { videoKey: "L21_V029", link: "https://youtube.com/watch?v=abc123", timestamp: 100 };
  const noOffset = buildVideoPlayback(item, 0);
  assert.equal(noOffset.type, "youtube");
  assert.equal(noOffset.start, 100);

  const withOffset = buildVideoPlayback(item, -172);
  assert.equal(withOffset.start, 0); // clamped at 0, not a magic L21_V029 constant
});
