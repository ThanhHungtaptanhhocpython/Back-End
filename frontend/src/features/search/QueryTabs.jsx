import { CloseOutlined, PlusOutlined } from "@ant-design/icons";

const renameBufRef = { current: "" };

export default function QueryTabs({ tabs, activeKey, onSelect, onAdd, onClose, onRename, editingKey, setEditingKey }) {
  const commitRename = (key) => {
    const v = renameBufRef.current.trim();
    if (v) onRename(key, v);
    setEditingKey(null);
  };
  return (
    <div className="ws-tabs">
      {tabs.map((t, i) =>
        editingKey === t.key ? (
          <input
            key={t.key}
            className="ws-tab-edit"
            autoFocus
            defaultValue={t.label}
            ref={renameBufRef}
            onBlur={() => commitRename(t.key)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename(t.key);
              if (e.key === "Escape") setEditingKey(null);
            }}
          />
        ) : (
          <div
            key={t.key}
            className={`ws-tab ${t.key === activeKey ? "active" : ""}`}
            onClick={() => onSelect(t.key)}
            onDoubleClick={() => setEditingKey(t.key)}
          >
            <span className="ws-tab-idx">{String(i + 1).padStart(2, "0")}</span>
            <span className={`ws-tab-status ${t.status}`} />
            <span>{t.label}</span>
            <span className="ws-tab-count">{t.total}</span>
            <button className="ws-tab-close" onClick={(e) => { e.stopPropagation(); onClose(t.key); }} title="Close tab">
              <CloseOutlined />
            </button>
          </div>
        )
      )}
      <button className="ws-tab-add" onClick={onAdd} title="New query tab">
        <PlusOutlined /> New query
      </button>
    </div>
  );
}
