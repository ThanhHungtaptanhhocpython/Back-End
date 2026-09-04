import assert from "node:assert/strict";
import test from "node:test";

import {
  shouldShowSetupNotice,
  summarizeReadiness,
} from "../src/features/workspace/setupNotice.js";

test("shows on a first run when readiness is unknown", () => {
  assert.equal(shouldShowSetupNotice({ dismissed: false, readiness: null }), true);
  assert.equal(shouldShowSetupNotice({}), true);
});

test("never shows once dismissed", () => {
  assert.equal(shouldShowSetupNotice({ dismissed: true, readiness: null }), false);
  assert.equal(shouldShowSetupNotice({ dismissed: true, readiness: { ok: false } }), false);
});

test("hides once readiness is confirmed OK", () => {
  assert.equal(shouldShowSetupNotice({ dismissed: false, readiness: { ok: true } }), false);
});

test("still shows while something is missing", () => {
  assert.equal(shouldShowSetupNotice({ dismissed: false, readiness: { ok: false } }), true);
});

test("summarizeReadiness flags a missing check as a warning", () => {
  const s = summarizeReadiness({
    ok: false,
    checks: [
      { id: "gpu", label: "torch / GPU", status: "ok" },
      { id: "model", label: "Jina CLIP v2 model", status: "miss" },
      { id: "index", label: "FAISS index + parquet", status: "miss" },
    ],
  });
  assert.equal(s.severity, "warning");
  assert.deepEqual(s.missing, ["Jina CLIP v2 model", "FAISS index + parquet"]);
  assert.match(s.headline, /still needs downloading/);
});

test("summarizeReadiness treats warn-only as info", () => {
  const s = summarizeReadiness({
    ok: true,
    checks: [{ id: "gpu", label: "torch / GPU", status: "warn" }],
  });
  assert.equal(s.severity, "info");
  assert.deepEqual(s.missing, ["torch / GPU"]);
});

test("summarizeReadiness with no payload gives the generic first-run headline", () => {
  const s = summarizeReadiness(null);
  assert.equal(s.severity, "info");
  assert.deepEqual(s.missing, []);
  assert.match(s.headline, /First run/);
});
