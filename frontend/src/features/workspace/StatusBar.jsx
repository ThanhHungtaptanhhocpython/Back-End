import { ReloadOutlined } from "@ant-design/icons";

export default function StatusBar({ clock, backend, onPing, onShortcuts }) {
  const isDemo = backend.demo !== false;
  return (
    <header className="ws-topbar">
      <div className="ws-brand">
        <div className="ws-brand-mark">
          <span>RK</span>
        </div>
        <div className="ws-brand-txt">
          <div className="ws-brand-title">
            Keyframe Retrieval <em>Workstation</em>
          </div>
          <div className="ws-brand-sub">AI Challenge 2026 - Multimodal video index - preview</div>
        </div>
      </div>

      <div className="ws-topsep" />

      <div className="ws-status-panel">
        <div className="ws-status-cell">
          <div className="ws-status-cell-label">Backend</div>
          <div className="ws-status-line">
            <span className={`ws-led ${backend.checking ? "check" : backend.backend === "online" ? "on" : "off"}`} />
            <span style={{ color: backend.backend === "online" ? "var(--ws-green)" : "var(--ws-red)" }}>
              {backend.checking ? "PROBING..." : backend.backend === "online" ? "ONLINE" : "OFFLINE"}
            </span>
            <button className="ws-status-btn" onClick={onPing}>
              <ReloadOutlined /> ping
            </button>
          </div>
        </div>
        <div className="ws-status-cell">
          <div className="ws-status-cell-label">Engine</div>
          <div className="ws-status-line">
            <span className={`ws-led ${isDemo ? "demo" : "on"}`} />
            <span style={{ color: isDemo ? "var(--ws-amber)" : "var(--ws-green)" }}>
              {isDemo ? "DEMO" : "LIVE"} - {backend.note || "LOCAL MOCK"}
            </span>
          </div>
        </div>
        {backend.at ? (
          <div className="ws-status-cell">
            <div className="ws-status-cell-label">Last ping</div>
            <div className="ws-status-line ws-dim" style={{ fontSize: "10px" }}>
              {backend.at}
            </div>
          </div>
        ) : null}
      </div>

      <div className="ws-topright">
        <button className="ws-status-btn" onClick={onShortcuts} title="Keyboard shortcuts">
          Shortcuts
        </button>

        <div className="ws-clock">
          <div className="ws-clock-time">{clock.time}</div>
          <div className="ws-clock-date">{clock.date}</div>
        </div>
      </div>
    </header>
  );
}
