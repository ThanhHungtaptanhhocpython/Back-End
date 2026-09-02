import assert from "node:assert/strict";
import test from "node:test";

import {
  progressHadErrors,
  shouldKeepPolling,
  summarizeSyncProgress,
} from "../src/features/settings/cloudSyncView.js";

test("a clean promoted run reads as success", () => {
  const v = summarizeSyncProgress({
    state: "done",
    version: "jina-v9",
    promoted: true,
    had_errors: false,
    artifacts: [{ name: "jina_faiss_index", status: "synced" }],
  });
  assert.equal(v.hadErrors, false);
  assert.equal(v.overallStatus, "success");
  assert.equal(v.tone, "success");
  assert.match(v.headline, /promoted/);
});

test("a run with a checksum failure is completed-with-errors, not green", () => {
  const prog = {
    state: "done",
    version: "jina-v9",
    promoted: false,
    had_errors: true,
    artifacts: [
      { name: "jina_faiss_index", status: "error", detail: "checksum mismatch" },
      { name: "jina_global_ids", status: "synced" },
    ],
  };
  assert.equal(progressHadErrors(prog), true);
  const v = summarizeSyncProgress(prog);
  assert.equal(v.overallStatus, "exception");
  assert.equal(v.tone, "warning");
  assert.match(v.headline, /completed with errors/i);
  assert.match(v.headline, /NOT promoted/);
});

test("errors are detected even when had_errors flag is absent", () => {
  assert.equal(
    progressHadErrors({ state: "done", artifacts: [{ status: "error" }] }),
    true,
  );
  assert.equal(
    progressHadErrors({ state: "done", report: { ok: false }, artifacts: [] }),
    true,
  );
  assert.equal(progressHadErrors({ state: "error", error: "boom" }), true);
});

test("a still-running sync keeps the poller alive and shows active", () => {
  const prog = { state: "running", version: "v1", trigger: "startup", artifacts: [] };
  assert.equal(shouldKeepPolling(prog), true);
  const v = summarizeSyncProgress(prog);
  assert.equal(v.running, true);
  assert.equal(v.overallStatus, "active");
  assert.match(v.headline, /startup/);
});

test("idle / done states stop the poller", () => {
  assert.equal(shouldKeepPolling({ state: "idle" }), false);
  assert.equal(shouldKeepPolling({ state: "done" }), false);
  assert.equal(shouldKeepPolling(null), false);
});

test("idle status hides the panel", () => {
  assert.equal(summarizeSyncProgress({ state: "idle" }).visible, false);
  assert.equal(summarizeSyncProgress(null).visible, false);
});
