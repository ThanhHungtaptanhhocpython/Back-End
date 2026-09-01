/**
 * Client for the loopback-only local management API (`/settings/...`).
 *
 * Every function accepts `{ fetchImpl }` so it can be unit-tested with
 * `node:test` (no DOM), mirroring `translateService.js`.
 */

import { getSearchConfig } from "./backendSearch.js";

export class SettingsApiError extends Error {
  constructor(message, { status = 0, body = null, kind = "response" } = {}) {
    super(message);
    this.name = "SettingsApiError";
    this.status = status;
    this.body = body;
    this.kind = kind;
  }
}

/** Base for management calls: the search base URL with any `/users` suffix removed. */
export function settingsBase(env) {
  const { baseUrl } = getSearchConfig(env);
  return String(baseUrl || "")
    .replace(/\/users$/i, "")
    .replace(/\/+$/, "");
}

function url(path, env) {
  const base = settingsBase(env);
  return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
}

async function request(method, path, { body, fetchImpl, env } = {}) {
  const impl = fetchImpl || globalThis.fetch;
  let response;
  try {
    response = await impl(url(path, env), {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    throw new SettingsApiError(`management API unreachable: ${error?.name || "network error"}`, {
      kind: "network",
    });
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.message)) || `HTTP ${response.status}`;
    throw new SettingsApiError(String(detail), {
      status: response.status,
      body: payload,
      kind: response.status === 403 ? "forbidden" : response.status === 409 ? "conflict" : "response",
    });
  }
  return payload ?? {};
}

// -- configuration ----------------------------------------------------------
export const fetchSchema = (o) => request("GET", "/settings/schema", o);
export const fetchConfig = (o) => request("GET", "/settings/config", o);
export const validateConfig = (payload, o = {}) =>
  request("POST", "/settings/validate", { ...o, body: payload });
export const saveConfig = (payload, o = {}) =>
  request("POST", "/settings/config", { ...o, body: payload });

// -- filesystem browser (loopback-only) --------------------------------
export function browseFs(path = "", { dirsOnly = false, showHidden = false, ...o } = {}) {
  const qs = new URLSearchParams();
  if (path) qs.set("path", path);
  if (dirsOnly) qs.set("dirs_only", "1");
  if (showHidden) qs.set("show_hidden", "1");
  const suffix = qs.toString() ? `?${qs}` : "";
  return request("GET", `/settings/fs${suffix}`, o);
}

// -- revisions ------------------------------------------------------------
export const fetchRevisions = (o) => request("GET", "/settings/revisions", o);
export const fetchRevision = (id, o) => request("GET", `/settings/revisions/${id}`, o);
export const restoreRevision = (id, o = {}) =>
  request("POST", `/settings/revisions/${id}/restore`, { ...o, body: {} });

// -- restart ------------------------------------------------------------
export const fetchRestartStatus = (o) => request("GET", "/settings/restart/status", o);
export const triggerRestart = (reason = "manual", o = {}) =>
  request("POST", "/settings/restart", { ...o, body: { reason } });

// -- AI providers -------------------------------------------------------
export const fetchProviders = (o) => request("GET", "/settings/providers", o);
export const testProvider = (id, mode = "text", o = {}) =>
  request("POST", `/settings/providers/${id}/test`, { ...o, body: { mode } });
export const discoverModels = (id, o) =>
  request("GET", `/settings/providers/${id}/models`, o);

// -- cloud assets ------------------------------------------------------
export const fetchCloudStatus = (o) => request("GET", "/settings/cloud/status", o);
export const testCloud = (o = {}) => request("POST", "/settings/cloud/test", { ...o, body: {} });
export const fetchCloudManifest = (refresh = false, o) =>
  request("GET", `/settings/cloud/manifest${refresh ? "?refresh=1" : ""}`, o);
export const syncCloud = (names = [], o = {}) =>
  request("POST", "/settings/cloud/sync", { ...o, body: { names } });
export const fetchCloudCache = (o) => request("GET", "/settings/cloud/cache", o);
export const clearCloudCache = (scope = "all", o = {}) =>
  request("POST", "/settings/cloud/cache/clear", { ...o, body: { scope } });
