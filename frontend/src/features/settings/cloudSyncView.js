/**
 * Pure view-model for the Cloud Assets sync panel.
 *
 * The backend `GET /settings/cloud/sync/status` payload must be turned into a
 * *truthful* UI state: a run that finished with checksum / download failures is
 * "completed with errors", never a green success, and a run that is still
 * downloading must keep the panel in its active state.
 */

const ERROR_ART_STATUSES = new Set(["error"]);

export function progressHadErrors(prog) {
  if (!prog || typeof prog !== "object") return false;
  if (prog.had_errors === true) return true;
  if (prog.state === "error") return true;
  if (typeof prog.error === "string" && prog.error.trim() !== "") return true;
  const arts = Array.isArray(prog.artifacts) ? prog.artifacts : [];
  if (arts.some((a) => ERROR_ART_STATUSES.has(a?.status))) return true;
  // an explicit report says it was not ok
  if (prog.report && prog.report.ok === false) return true;
  return false;
}

/**
 * @returns {{
 *   visible: boolean, running: boolean, done: boolean, hadErrors: boolean,
 *   promoted: boolean, overallStatus: "active"|"exception"|"success"|"normal",
 *   headline: string, tone: "info"|"warning"|"error"|"success"
 * }}
 */
export function summarizeSyncProgress(prog) {
  const state = prog?.state || "idle";
  const running = state === "running";
  const done = state === "done";
  const hadErrors = progressHadErrors(prog);
  const promoted = Boolean(prog?.promoted ?? prog?.report?.promoted);
  const version = prog?.version || "?";
  const fromStartup = prog?.trigger === "startup";

  let overallStatus = "normal";
  if (running) overallStatus = "active";
  else if (hadErrors) overallStatus = "exception";
  else if (done) overallStatus = "success";

  let headline;
  let tone;
  if (running) {
    headline = `Syncing ${fromStartup ? "(startup) " : ""}version ${version} …`;
    tone = "info";
  } else if (state === "error") {
    headline = `Sync failed: ${prog?.error || "unknown error"}`;
    tone = "error";
  } else if (hadErrors) {
    headline = `Sync completed with errors — version ${version} was NOT promoted`;
    tone = "warning";
  } else if (done) {
    headline = `Sync ${promoted ? "promoted " : "finished for "}version ${version}`;
    tone = "success";
  } else {
    headline = "No sync has run yet";
    tone = "info";
  }

  return {
    visible: state !== "idle",
    running,
    done,
    hadErrors,
    promoted,
    overallStatus,
    headline,
    tone,
  };
}

/** Should the poller keep running given the latest status payload? */
export function shouldKeepPolling(prog) {
  return (prog?.state || "idle") === "running";
}
