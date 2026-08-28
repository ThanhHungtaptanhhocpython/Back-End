import { useRef } from "react";
import { SearchOutlined, UploadOutlined, VideoCameraOutlined } from "@ant-design/icons";
import { SEARCH_TYPES } from "../../shared/constants";
import Keycap from "../../components/Keycap";

export default function SearchBar({ tab, onPatch, onRun, onAgentRun, searchRef, toast }) {
  const fileRef = useRef(null);
  const isImage = tab.searchType === "IMAGE";
  return (
    <div className="ws-searchbar">
      <div className="ws-panel-tag">
        <VideoCameraOutlined /> Query controls
      </div>

      <div className="ws-types">
        {SEARCH_TYPES.map((t) => (
          <button
            key={t.value}
            className={`ws-type ${tab.searchType === t.value ? "active" : ""}`}
            onClick={() => onPatch({ searchType: t.value })}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="ws-query-wrap">
        <input
          ref={searchRef}
          className="ws-query"
          placeholder={isImage ? "Select a reference frame to seed the search" : "e.g. 'cam 04', 'overpass', 'forklift'..."}
          value={tab.query}
          onChange={(e) => onPatch({ query: e.target.value })}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) onRun();
          }}
        />
        <button className="ws-query-run" onClick={onRun} title="Run search">
          <SearchOutlined />
        </button>
      </div>
      <div className="ws-query-hint">
        <span>{tab.status === "running" ? <span className="ws-cyan">Searching...</span> : tab.status === "done" ? <span className="ws-green">{tab.total} frames - {tab.latency}ms</span> : <span className="ws-faint">Ready</span>}</span>
        <span className="ws-hint-keys">
          <Keycap>Esc</Keycap> leave input
        </span>
      </div>

      {isImage && (
        <div>
          <div className="ws-panel-tag">Reference image</div>
          <div className={`ws-upload ${tab.params.imageFile ? "has" : ""}`} onClick={() => fileRef.current?.click()}>
            <UploadOutlined />
            {tab.params.imageFile ? `Seed: ${tab.params.imageFile.name || tab.params.imageFile}` : "Upload reference image"}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) {
                onPatch({ params: { ...tab.params, imageFile: f } });
                toast.info(`Reference image selected: ${f.name}`);
              }
            }}
          />
        </div>
      )}

      <div className="ws-params">
        <div className="ws-param">
          <label className="ws-param-label">Top K</label>
          <input
            className="ws-num"
            type="number"
            min={1}
            max={80}
            value={tab.params.topk}
            onChange={(e) => onPatch({ params: { ...tab.params, topk: Number(e.target.value) } })}
          />
        </div>
        <div className="ws-param">
          <label className="ws-param-label">Clip</label>
          <button className={`ws-switch ${tab.params.clip ? "on" : ""}`} onClick={() => onPatch({ params: { ...tab.params, clip: !tab.params.clip } })} />
        </div>
        <div className="ws-param">
          <label className="ws-param-label">CLIPv2</label>
          <button className={`ws-switch ${tab.params.clipv2 ? "on" : ""}`} onClick={() => onPatch({ params: { ...tab.params, clipv2: !tab.params.clipv2 } })} />
        </div>
      </div>

      <div className="ws-runbar">
        <button className="ws-btn primary" onClick={onRun} disabled={tab.status === "running"}>
          {tab.status === "running" ? "Searching..." : "Search"}
        </button>
        <button className="ws-btn" onClick={onAgentRun} disabled={tab.status === "running" || !String(tab.query || "").trim()} title="Run AI Query Coordinator">
          <SearchOutlined /> Agent Search
        </button>
      </div>
    </div>
  );
}
