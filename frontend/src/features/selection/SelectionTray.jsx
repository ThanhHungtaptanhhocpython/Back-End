import { CloseOutlined, ExportOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { keyframeImgProps } from "../../shared/imageFallback";

export default function SelectionTray({ keptItems, onRemove, onClear, onExport, onOpenBatch, onOpen, trayRef }) {
  return (
    <aside className="ws-tray" ref={trayRef}>
      <div className="ws-tray-head">
        <div className="ws-tray-title">
          <ExportOutlined /> Selection Tray
        </div>
        <div className="ws-tray-count">
          <b>{keptItems.length}</b> / kept - export-ready
        </div>
        <div className="ws-tray-spacer" />
        <div className="ws-tray-btns">
          <button className="ws-btn small" onClick={onClear} disabled={keptItems.length === 0}>
            Clear
          </button>
          <button className="ws-btn small" style={{ color: "#2563eb", borderColor: "#8fb2ff" }} onClick={onOpenBatch} title="Paste all queries and export one ZIP">
            <ThunderboltOutlined /> Batch Submit
          </button>
          <button className="ws-btn small export" onClick={onExport}>
            <ExportOutlined /> Export Submission
          </button>
        </div>
      </div>
      {keptItems.length === 0 ? (
        <div className="ws-tray-empty">Tray empty - press Space on a focused frame to keep it.</div>
      ) : (
        <div className="ws-tray-strip">
          {keptItems.map((item) => (
            <div key={item.id} className="ws-tray-thumb" onClick={() => onOpen(item)}>
              {item.image ? (
                <img src={item.image} alt={item.frameName} {...keyframeImgProps} />
              ) : (
                <div
                  className="ws-tray-thumb-missing"
                  title={item.previewError || "Exact preview unavailable — the frame is still export-ready."}
                >
                  Preview unavailable
                </div>
              )}
              <button className="ws-tray-x" onClick={(e) => { e.stopPropagation(); onRemove(item.id); }} title="Remove from tray">
                <CloseOutlined />
              </button>
              <div className="ws-tray-thumb-label">{item.frameName}</div>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

