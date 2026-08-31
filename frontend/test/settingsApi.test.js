import assert from "node:assert/strict";
import test from "node:test";

import {
  SettingsApiError,
  fetchSchema,
  saveConfig,
  settingsBase,
  syncCloud,
  testProvider,
  triggerRestart,
} from "../src/services/settingsApi.js";

const ENV = { VITE_SEARCH_API_BASE_URL: "http://127.0.0.1:3000/users" };

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body };
}

test("settingsBase strips the /users suffix and trailing slashes", () => {
  assert.equal(settingsBase(ENV), "http://127.0.0.1:3000");
  assert.equal(settingsBase({ VITE_SEARCH_API_BASE_URL: "http://x:1/" }), "http://x:1");
  assert.equal(settingsBase({}), "");
});

test("fetchSchema GETs the schema endpoint", async () => {
  let seen;
  const body = await fetchSchema({
    env: ENV,
    fetchImpl: async (url, init) => {
      seen = { url, method: init?.method };
      return jsonResponse({ groups: [] });
    },
  });
  assert.equal(seen.url, "http://127.0.0.1:3000/settings/schema");
  assert.equal(seen.method, "GET");
  assert.deepEqual(body, { groups: [] });
});

test("saveConfig POSTs the full payload", async () => {
  let init;
  const payload = { values: { PORT: "4000" }, secret_set: { OPENROUTER_API_KEY: "sk" }, secret_clear: [] };
  await saveConfig(payload, {
    env: ENV,
    fetchImpl: async (url, _init) => {
      init = _init;
      assert.equal(url, "http://127.0.0.1:3000/settings/config");
      return jsonResponse({ ok: true, revision_id: 7 });
    },
  });
  assert.equal(init.method, "POST");
  assert.deepEqual(JSON.parse(init.body), payload);
});

test("a 409 becomes a conflict SettingsApiError carrying the body", async () => {
  await assert.rejects(
    saveConfig(
      { values: {} },
      { env: ENV, fetchImpl: async () => jsonResponse({ detail: "store disabled" }, { ok: false, status: 409 }) },
    ),
    (err) => {
      assert.ok(err instanceof SettingsApiError);
      assert.equal(err.status, 409);
      assert.equal(err.kind, "conflict");
      assert.equal(err.message, "store disabled");
      return true;
    },
  );
});

test("a 403 is surfaced as kind=forbidden", async () => {
  await assert.rejects(
    fetchSchema({ env: ENV, fetchImpl: async () => jsonResponse({ detail: "loopback only" }, { ok: false, status: 403 }) }),
    (err) => err.kind === "forbidden",
  );
});

test("a network throw becomes kind=network", async () => {
  await assert.rejects(
    fetchSchema({
      env: ENV,
      fetchImpl: async () => {
        throw new TypeError("Failed to fetch");
      },
    }),
    (err) => err instanceof SettingsApiError && err.kind === "network",
  );
});

test("testProvider posts the mode to the provider test endpoint", async () => {
  let seen;
  await testProvider("groq", "vision", {
    env: ENV,
    fetchImpl: async (url, init) => {
      seen = { url, body: JSON.parse(init.body) };
      return jsonResponse({ ok: true });
    },
  });
  assert.equal(seen.url, "http://127.0.0.1:3000/settings/providers/groq/test");
  assert.deepEqual(seen.body, { mode: "vision" });
});

test("syncCloud posts the requested artifact names", async () => {
  let body;
  await syncCloud(["faiss_index"], {
    env: ENV,
    fetchImpl: async (_url, init) => {
      body = JSON.parse(init.body);
      return jsonResponse({ ok: true, promoted: false });
    },
  });
  assert.deepEqual(body, { names: ["faiss_index"] });
});

test("triggerRestart posts a reason", async () => {
  let body;
  await triggerRestart("manual", {
    env: ENV,
    fetchImpl: async (_url, init) => {
      body = JSON.parse(init.body);
      return jsonResponse({ ok: false, detail: "launcher_not_running" });
    },
  });
  assert.deepEqual(body, { reason: "manual" });
});
