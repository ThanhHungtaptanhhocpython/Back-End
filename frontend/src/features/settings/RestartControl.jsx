import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Popconfirm, Tag, Tooltip, message } from "antd";

import { fetchRestartStatus, triggerRestart } from "../../services/settingsApi.js";
import { RESTART_STATE_LABEL } from "./settingsModel.js";

const TRANSIENT = new Set(["restarting", "polling-health", "rolling-back"]);
const COLOR = {
  healthy: "green",
  "rollback-complete": "orange",
  failed: "red",
  idle: "default",
};

export default function RestartControl({ pendingRevision }) {
  const [status, setStatus] = useState(null);
  const timer = useRef(null);

  const poll = useCallback(async () => {
    try {
      setStatus(await fetchRestartStatus());
    } catch {
      setStatus((s) => s || { state: "idle", launcher_running: false });
    }
  }, []);

  useEffect(() => {
    poll();
    timer.current = setInterval(poll, 2000);
    return () => clearInterval(timer.current);
  }, [poll]);

  async function onRestart() {
    try {
      const res = await triggerRestart("manual");
      if (res.ok) message.info("Restart requested — the launcher is applying it.");
      else message.warning("No launcher is running. Restart the app manually to apply changes.");
      poll();
    } catch (err) {
      message.error(err.message);
    }
  }

  const state = status?.state || "idle";
  const running = Boolean(status?.launcher_running);
  const busy = TRANSIENT.has(state);

  return (
    <div className="set-restart">
      <Tooltip
        title={
          running
            ? "The local launcher is running and will apply restarts automatically."
            : "No launcher heartbeat. Start with `python -m launcher` for one-click restarts."
        }
      >
        <Tag color={running ? "blue" : "default"}>{running ? "launcher live" : "no launcher"}</Tag>
      </Tooltip>
      <Tag color={busy ? "processing" : COLOR[state] || "default"}>
        {RESTART_STATE_LABEL[state] || state}
      </Tag>
      {status?.failed_revision_id != null && state !== "healthy" && (
        <Tag color="red">rolled back from #{status.failed_revision_id}</Tag>
      )}
      {pendingRevision != null && (
        <Tag color="orange">revision {pendingRevision} pending</Tag>
      )}
      <Popconfirm
        title="Restart now?"
        description="The app will be briefly unavailable while it restarts. If the new configuration does not come up healthy, the launcher restores the previous revision automatically."
        okText="Restart"
        onConfirm={onRestart}
      >
        <Button danger loading={busy}>Restart now</Button>
      </Popconfirm>
    </div>
  );
}
