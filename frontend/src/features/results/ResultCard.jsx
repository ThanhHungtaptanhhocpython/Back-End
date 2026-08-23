import { DeleteOutlined, PlusSquareOutlined } from "@ant-design/icons";
import { fmtDur } from "../../shared/format";

export default function ResultCard({ item, focused, isKept, onOpen, onToggleKeep, onExclude, onPivot, onFocusItem, registerRef }) {
  return (
    <div
      className={`ws-card ${focused ? "focused" : ""}`}
      tabIndex={focused ? 0 : -1}
      data-wscard="1"
      role="gridcell"
      aria-label={`Frame ${item.frameName} - ${Math.round(item.score * 100)}% match`}
      ref={(el) => registerRef(item.id, el)}
      onFocus={() => onFocusItem(item.id)}
      onClick={(event) => {
        event.currentTarget.focus({ preventScroll: true });
        onOpen(item);
      }}
    >
      <div className="ws-thumb">
        <img src={item.image} alt={item.frameName} loading="lazy" />
        <span className="ws-rank">{String(item.rank).padStart(2, "0")}</span>
        <span className="ws-score">{Math.round(item.score * 100)}%</span>
        {item.real ? <span className="ws-live">LIVE FEED</span> : null}
        <span className="ws-tc">{item.timecode}</span>
      </div>
      <div className="ws-card-meta">
        <div className="ws-meta-row">
          <span className="k">Video</span>
          <span className="v cyan" title={item.videoKey}>{item.videoKey}</span>
        </div>
        <div className="ws-meta-row">
          <span className="k">Frame</span>
          <span className="v" title={item.frameName}>{item.frameName}</span>
        </div>
        <div className="ws-meta-row">
          <span className="k">Time</span>
          <span className="v">{fmtDur(item.timestamp)} - {item.folderKey}</span>
        </div>
      </div>
      <div className="ws-card-actions">
        <button
          className="ws-card-act sim"
          onClick={(e) => { e.stopPropagation(); onPivot(item); }}
          title="Similar-image pivot"
        >
          <PlusSquareOutlined /> Similar
        </button>
        <button
          className={`ws-card-act keep ${isKept ? "on" : ""}`}
          onClick={(e) => { e.stopPropagation(); onToggleKeep(item); }}
          title="Toggle selection tray (Space)"
        >
          {isKept ? "Kept" : "Keep"}
        </button>
        <button
          className="ws-card-act del"
          onClick={(e) => { e.stopPropagation(); onExclude(item); }}
          title="Remove from results"
        >
          <DeleteOutlined /> Remove
        </button>
      </div>
    </div>
  );
}
