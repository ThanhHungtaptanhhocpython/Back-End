import { useRef } from "react";
import { CloseOutlined, QuestionCircleOutlined } from "@ant-design/icons";
import Keycap from "../../components/Keycap";
import useDialogFocus from "../../hooks/useDialogFocus";

const ALWAYS_ROWS = [
  ["Open help / keyboard map", [["?"]]],
  ["Move focus in grid", [["↑"], ["↓"], ["←"], ["→"]]],
  ["Go to first / last frame", [["Home"], ["End"]]],
  ["Open frame review", [["↵"]]],
  ["Keep / unkeep frame", [["␣"]]],
  ["Remove frame (with undo)", [["Del"]]],
  ["Reviewer: previous / next frame", [["←"], ["→"]]],
  ["Reviewer: keep / remove", [["␣"], ["Del"]]],
  ["Leave field, then close overlay (layered)", [["Esc"]]],
];

const POWER_ROWS = [
  ["Focus query field", [["/"]]],
  ["Similar-image pivot", [["S"]]],
  ["Remove frame (with undo)", [["X"]]],
  ["Export submission", [["E"]]],
];

function RowList({ rows }) {
  return (
    <div className="ws-shortcuts">
      {rows.map(([label, keyGroups]) => (
        <div key={label} className="ws-sc-row">
          <span>{label}</span>
          <span className="ws-sc-keys">
            {keyGroups.map((group, gi) => (
              <span key={gi} style={{ display: "flex", gap: 4, marginRight: 4 }}>
                {group.map((k, i) => <Keycap key={i}>{k}</Keycap>)}
              </span>
            ))}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function ShortcutOverlay({ open, powerUser, onTogglePowerUser, onClose }) {
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);
  if (!open) return null;
  return (
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="ws-modal" role="dialog" aria-modal="true" aria-label="Keyboard map" tabIndex={-1} style={{ width: "min(760px, 96vw)" }}>
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <QuestionCircleOutlined /> Keyboard Map
          </div>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Close (Esc)">
            <CloseOutlined />
          </button>
        </div>
        <div className="ws-modal-body">
          <div className="ws-kbd-switch">
            <span className="ws-kbd-switch-label">Power-user shortcuts</span>
            <span className="ws-kbd-switch-sub">Optional accelerators — off by default</span>
            <button
              className={`ws-switch ${powerUser ? "on" : ""}`}
              onClick={onTogglePowerUser}
              role="switch"
              aria-checked={powerUser}
              title="Turn power-user shortcuts on or off"
            />
          </div>

          <div className="ws-shortcuts-group">
            <div className="ws-shortcuts-group-title">Always available</div>
            <RowList rows={ALWAYS_ROWS} />
          </div>

          <div className="ws-shortcuts-group">
            <div className="ws-shortcuts-group-title">Power-user mode {powerUser ? "on" : "(off)"}</div>
            <RowList rows={POWER_ROWS} />
          </div>

          <p className="ws-shortcuts-note">
            Browser-reserved shortcuts (Ctrl/Cmd+W, T, L, R, N, ⌘↵) are intentionally avoided — the workstation relies
            on plain, context-scoped keys. Keys are ignored while typing or during IME composition, modifiers are never
            intercepted, and removal uses <b>Delete</b> (never Backspace) with an Undo chip.
          </p>
        </div>
      </div>
    </div>
  );
}
