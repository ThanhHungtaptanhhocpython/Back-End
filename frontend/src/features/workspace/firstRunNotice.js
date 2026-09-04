/**
 * Logic for the one-time "you still need to download the model + index" notice
 * shown on the Workstation. Kept DOM-free so it is unit-testable with
 * `node:test` (see test/firstRunNotice.test.js).
 */

export const SETUP_NOTICE_KEY = "hcmai.setupNoticeDismissed";
export const CLOUD_ASSETS_HASH = "#/settings/cloud";

/**
 * Should the setup notice be visible?
 *  - never once the user has dismissed it
 *  - never once readiness is confirmed OK (nothing left to download)
 *  - otherwise yes (first run, or something is still missing)
 *
 * @param {{dismissed?: boolean, readiness?: {ok?: boolean}|null}} opts
 */
export function shouldShowSetupNotice({ dismissed = false, readiness = null } = {}) {
  if (dismissed) return false;
  if (readiness && readiness.ok === true) return false;
  return true;
}

/**
 * Turn a /settings/jina/readiness payload into { severity, missing, headline }.
 * `severity` drives the antd Alert type; `missing` lists the check labels that
 * are not yet satisfied.
 */
export function summarizeReadiness(readiness) {
  const checks = readiness && Array.isArray(readiness.checks) ? readiness.checks : [];
  const miss = checks.filter((c) => c && c.status === "miss").map((c) => c.label);
  const warn = checks.filter((c) => c && c.status === "warn").map((c) => c.label);

  if (miss.length) {
    return {
      severity: "warning",
      missing: miss,
      headline: `Not ready yet — ${miss.join(", ")} still needs downloading.`,
    };
  }
  if (warn.length) {
    return {
      severity: "info",
      missing: warn,
      headline: `Mostly ready — ${warn.join(", ")} could be improved.`,
    };
  }
  return {
    severity: "info",
    missing: [],
    headline:
      "First run: the Jina CLIP v2 model and search index download on first use.",
  };
}
