import { VideoCameraOutlined } from "@ant-design/icons";
import ResultCard from "./ResultCard";

export default function ResultGrid({ tab, keptMap, focusedId, onFocusItem, onOpen, onToggleKeep, onExclude, onPivot, registerRef }) {
  if (tab.status === "running") {
    return (
      <div className="ws-grid" aria-label="Searching local mock index">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="ws-skeleton-card">
            <div className="ws-skeleton-thumb" />
            <div className="ws-skeleton-line w1" />
            <div className="ws-skeleton-line w2" />
          </div>
        ))}
      </div>
    );
  }
  if (tab.results.length === 0) {
    return (
      <div className="ws-empty">
        <VideoCameraOutlined style={{ fontSize: 34, color: "#94a3b8" }} />
        <div>No frames matched this query.</div>
        <div className="ws-empty-sub">Try different wording, then run the search.</div>
      </div>
    );
  }
  const activeId = tab.results.some((r) => r.id === focusedId) ? focusedId : tab.results[0]?.id ?? null;
  return (
    <div className="ws-grid" role="grid" aria-label={`Results grid - ${tab.results.length} frames`}>
      {tab.results.map((item) => (
        <ResultCard
          key={item.id}
          item={item}
          focused={activeId === item.id}
          isKept={keptMap.has(item.id)}
          registerRef={registerRef}
          onFocusItem={onFocusItem}
          onOpen={onOpen}
          onToggleKeep={onToggleKeep}
          onExclude={onExclude}
          onPivot={onPivot}
        />
      ))}
    </div>
  );
}
