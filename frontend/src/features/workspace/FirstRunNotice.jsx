import { useEffect, useState } from "react";
import { Alert, Button } from "antd";

import { fetchJinaReadiness } from "../../services/settingsApi.js";
import {
  CLOUD_ASSETS_HASH,
  SETUP_NOTICE_KEY,
  shouldShowSetupNotice,
  summarizeReadiness,
} from "./setupNotice.js";

function readDismissed() {
  try {
    return window.localStorage.getItem(SETUP_NOTICE_KEY) === "1";
  } catch {
    return false;
  }
}

function persistDismissed() {
  try {
    window.localStorage.setItem(SETUP_NOTICE_KEY, "1");
  } catch {
    /* private mode / storage disabled — the notice just reappears next load */
  }
}

/**
 * One-time banner on the Workstation pointing first-time users at
 * Settings → Cloud Assets, where the Jina CLIP v2 model + FAISS index are
 * downloaded. Hides itself once dismissed, or once readiness reports OK.
 */
export default function FirstRunNotice() {
  const [dismissed, setDismissed] = useState(readDismissed);
  const [readiness, setReadiness] = useState(null);

  useEffect(() => {
    if (dismissed) return undefined;
    let alive = true;
    fetchJinaReadiness()
      .then((r) => {
        if (!alive) return;
        setReadiness(r);
        if (r && r.ok === true) persistDismissed(); // nothing to do — don't nag again
      })
      .catch(() => {
        /* backend not up yet: still show the generic first-run hint */
      });
    return () => {
      alive = false;
    };
  }, [dismissed]);

  if (!shouldShowSetupNotice({ dismissed, readiness })) return null;

  const { severity, headline } = summarizeReadiness(readiness);

  function dismiss() {
    persistDismissed();
    setDismissed(true);
  }

  return (
    <Alert
      banner
      closable
      type={severity}
      onClose={dismiss}
      style={{ borderRadius: 0 }}
      message={
        <span>
          {headline}{" "}
          <Button
            size="small"
            type="link"
            style={{ padding: 0, height: "auto" }}
            onClick={() => {
              window.location.hash = CLOUD_ASSETS_HASH;
            }}
          >
            Open Settings → Cloud Assets
          </Button>{" "}
          to check readiness and download.
        </span>
      }
    />
  );
}
