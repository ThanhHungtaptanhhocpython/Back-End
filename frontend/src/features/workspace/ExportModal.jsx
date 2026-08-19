import { useRef, useState } from "react";
import { CloseOutlined, ExportOutlined } from "@ant-design/icons";
import useDialogFocus from "../../hooks/useDialogFocus";

export default function ExportModal({ open, items, onClose, toast }) {
  const [format, setFormat] = useState("CSV");
  const [name, setName] = useState("");
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);
  if (!open) return null;

  const preview =
    format === "CSV"
      ? "folder_key,global_frame_id,frame_name\n" + items.slice(0, 5).map((i) => `${i.folderKey},${i.globalFrameId},${i.frameName}`).join("\n") + (items.length > 5 ? `\n… +${items.length - 5} more` : "")
      : JSON.stringify(items.slice(0, 2).map((i) => ({ ...i, image: "[…]" })), null, 2) + (items.length > 2 ? `\n… +${items.length - 2} more` : "");

  const download = () => {
    if (items.length === 0) {
      toast.warning("Nothing to export");
      return;
    }
    let content;
    let ext;
    if (format === "CSV") {
      ext = "csv";
      content = "folder_key,global_frame_id,frame_name\n" + items.map((i) => `${i.folderKey},${i.globalFrameId},${i.frameName}`).join("\n");
    } else {
      ext = "json";
      content = JSON.stringify(items.map((item) => {
        const exportedItem = { ...item };
        delete exportedItem.image;
        return exportedItem;
      }), null, 2);
    }
    const blob = new Blob([content], { type: `text/${ext === "csv" ? "csv" : "json"};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(name.trim() || "submission").replace(/[\\/:*?"<>|]+/g, "_")}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${items.length} frames as ${format.toUpperCase()}`);
    onClose();
  };

  return (
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="ws-modal" role="dialog" aria-modal="true" aria-label="Export submission" tabIndex={-1}>
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <ExportOutlined /> Export Submission
          </div>
          <span className="ws-dim" style={{ fontSize: "11px", letterSpacing: "1px" }}>
            {items.length} FRAMES · {items.length === 0 ? "TRAY EMPTY" : "READY"}
          </span>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Close (Esc)">
            <CloseOutlined />
          </button>
        </div>
        <div className="ws-modal-body">
          <div className="ws-exp-fmt">
            {["CSV", "JSON"].map((f) => (
              <button key={f} className={`ws-type ${format === f ? "active" : ""}`} style={{ minWidth: 120 }} onClick={() => setFormat(f)}>
                {f === "CSV" ? "CSV (submission format)" : "JSON (full metadata)"}
              </button>
            ))}
          </div>
          <div className="ws-field">
            <label className="ws-field-label">Filename</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="submission" />
          </div>
          <div className="ws-field">
            <label className="ws-field-label">Preview</label>
            <div className="ws-exp-preview">{preview}</div>
          </div>
          <div className="ws-runbar">
            <button className="ws-btn primary" onClick={download} disabled={items.length === 0}>
              <ExportOutlined /> Download
            </button>
            <button className="ws-btn" onClick={onClose}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
