import { useRef, useState } from "react";
import { QuestionCircleOutlined, SearchOutlined, UploadOutlined, VideoCameraOutlined } from "@ant-design/icons";
import { SEARCH_TYPES } from "../../shared/constants";
import Keycap from "../../components/Keycap";
import TemporalEditor from "./TemporalEditor";
import SearchGuide from "./SearchGuide";

export default function SearchBar({ tab, onPatch, onRun, onAgentRun, searchRef, toast }) {
  const fileRef = useRef(null);
  const [guideOpen, setGuideOpen] = useState(false);
  const isImage = tab.searchType === "IMAGE";
  const isTemporal = tab.searchType === "TEMPORAL";
  return (
    <div className="ws-searchbar">
      <div className="ws-searchbar-head">
        <div className="ws-panel-tag">
          <VideoCameraOutlined /> Query controls
        </div>
        <button
          type="button"
          className="ws-guide-open"
          onClick={() => setGuideOpen(true)}
          title="Chế độ nào cho việc gì? (Q&A vs Agent Search...)"
        >
          <QuestionCircleOutlined /> Dùng chế độ nào?
        </button>
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

      <SearchGuide open={guideOpen} onClose={() => setGuideOpen(false)} />

      {isTemporal ? (
        <>
          <TemporalEditor tab={tab} onPatch={onPatch} onRun={onRun} />
          <div className="ws-runbar">
            <button className="ws-btn" onClick={onAgentRun} disabled={tab.status === "running" || !String(tab.query || "").trim()} title="Run AI Query Coordinator">
              <SearchOutlined /> Agent Search
            </button>
          </div>
        </>
      ) : (
      <>
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
      </div>

      <div className="ws-runbar">
        <button className="ws-btn primary" onClick={onRun} disabled={tab.status === "running"}>
          {tab.status === "running" ? "Searching..." : "Search"}
        </button>
        <button className="ws-btn" onClick={onAgentRun} disabled={tab.status === "running" || !String(tab.query || "").trim()} title="Run AI Query Coordinator">
          <SearchOutlined /> Agent Search
        </button>
      </div>
      </>
      )}
    </div>
  );
}
