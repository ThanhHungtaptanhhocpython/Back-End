import { useState, useRef } from "react";
import { 
  CloseOutlined, 
  ThunderboltOutlined, 
  UploadOutlined, 
  DownloadOutlined, 
  CheckCircleOutlined,
  LoadingOutlined
} from "@ant-design/icons";
import useDialogFocus from "../../hooks/useDialogFocus";
import { makeSubmissionZip, buildSubmissionCsv, sanitizeQueryFileName } from "../../shared/submissionExport";
import { runSearch } from "../../shared/adapters";

export default function BatchQueryModal({ open, onClose, toast }) {
  const [inputText, setInputText] = useState("");
  const [parsedQueries, setParsedQueries] = useState([]);
  const [topk, setTopk] = useState(100);
  const [zipName, setZipName] = useState("submission.zip");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, status: "" });
  const [resultsSummary, setResultsSummary] = useState(null);
  
  const fileInputRef = useRef(null);
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);

  if (!open) return null;

  // Parse queries from textarea input
  const handleParseText = (text) => {
    setInputText(text);
    if (!text.trim()) {
      setParsedQueries([]);
      return;
    }

    const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
    const queries = [];
    let currentQuery = null;

    lines.forEach((line) => {
      // Check if line is a header like 'query-p1-1-kis:' or '1.' or 'Q1:'
      const matchHeader = line.match(/^(?:query-p1-(\d+)-(kis|qa|trake)|query-(\d+)-(kis|qa|trake)|(\d+)[.:)]|Q(\d+)[.:)])\s*:?\s*(.*)$/i);
      
      if (matchHeader) {
        const qNum = matchHeader[1] || matchHeader[3] || matchHeader[5] || matchHeader[6] || String(queries.length + 1);
        const qType = (matchHeader[2] || matchHeader[4] || "kis").toLowerCase();
        const qContent = matchHeader[7] || "";
        
        currentQuery = {
          name: `query-p1-${qNum}-${qType}.csv`,
          type: qType,
          text: qContent,
        };
        queries.push(currentQuery);
      } else if (currentQuery && currentQuery.text) {
        // Append multi-line query
        currentQuery.text += " " + line;
      } else {
        // Line without specific header
        const qNum = queries.length + 1;
        currentQuery = {
          name: `query-p1-${qNum}-kis.csv`,
          type: "kis",
          text: line,
        };
        queries.push(currentQuery);
      }
    });

    setParsedQueries(queries.filter((q) => q.text.trim()));
  };

  // Handle uploading multiple .txt files
  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    const filePromises = files.map((file) => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
          const content = String(event.target?.result || "").trim();
          const stem = file.name.replace(/\.(txt|csv)$/i, "");
          const matchType = stem.match(/-(kis|qa|trake)$/i);
          const type = matchType ? matchType[1].toLowerCase() : "kis";
          resolve({
            name: `${stem}.csv`,
            type,
            text: content,
          });
        };
        reader.readAsText(file);
      });
    });

    Promise.all(filePromises).then((loaded) => {
      // Sort naturally by query number
      loaded.sort((a, b) => {
        const numA = parseInt((a.name.match(/-(\d+)/) || [])[1] || "999", 10);
        const numB = parseInt((b.name.match(/-(\d+)/) || [])[1] || "999", 10);
        return numA - numB;
      });
      setParsedQueries(loaded);
      const textSummary = loaded.map((q) => `${q.name.replace('.csv','')}: ${q.text}`).join("\n\n");
      setInputText(textSummary);
      toast.success(`Loaded ${loaded.length} query files successfully!`);
    });
  };

  // Run batch search & download zip
  const handleRunBatch = async () => {
    if (!parsedQueries.length) {
      toast.warning("Please paste or upload at least one query.");
      return;
    }

    setRunning(true);
    setResultsSummary(null);
    const total = parsedQueries.length;
    const generatedFiles = [];
    const summaryItems = [];

    for (let i = 0; i < total; i++) {
      const q = parsedQueries[i];
      setProgress({
        current: i + 1,
        total,
        status: `[${i + 1}/${total}] Searching: ${q.name.replace('.csv','')}...`,
      });

      try {
        const searchTab = {
          key: `batch_${i}`,
          searchType: q.type.toUpperCase() === "QA" ? "QA" : "TEXT",
          query: q.text,
          params: { topk },
        };

        const res = await runSearch(searchTab);
        const items = (res?.items || []).slice(0, topk);
        const csvContent = buildSubmissionCsv(items, q.type);
        
        generatedFiles.push({
          name: sanitizeQueryFileName(q.name, q.type),
          content: csvContent,
          queryType: q.type,
        });

        summaryItems.push({
          name: q.name,
          type: q.type,
          count: items.length,
          preview: buildSubmissionCsv(items.slice(0, 2), q.type).split("\n").join(" | "),
        });
      } catch (err) {
        console.error(`Error on query ${q.name}:`, err);
        // Fallback empty CSV
        generatedFiles.push({
          name: sanitizeQueryFileName(q.name, q.type),
          content: "",
          queryType: q.type,
        });
        summaryItems.push({
          name: q.name,
          type: q.type,
          count: 0,
          preview: "Failed: " + err.message,
        });
      }
    }

    setProgress({ current: total, total, status: "Packing ZIP archive..." });

    // Pack into root-level ZIP
    const blob = makeSubmissionZip(generatedFiles);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeZipName = (zipName.trim() || "submission.zip").replace(/[\\/:*?"<>|]+/g, "_");
    a.download = safeZipName.toLowerCase().endsWith(".zip") ? safeZipName : `${safeZipName}.zip`;
    a.click();
    URL.revokeObjectURL(url);

    setRunning(false);
    setResultsSummary(summaryItems);
    toast.success(`Batch execution complete! Downloaded '${a.download}' with ${total} query CSVs.`);
  };

  return (
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget && !running) onClose(); }}>
      <div 
        ref={dialogRef} 
        className="ws-modal" 
        style={{ maxWidth: 780, width: "95%" }}
        role="dialog" 
        aria-modal="true" 
        tabIndex={-1}
      >
        <div className="ws-modal-head">
          <div className="ws-modal-title" style={{ color: "#58a6ff" }}>
            <ThunderboltOutlined /> Batch Query Submitter (Multi-Query ZIP Engine)
          </div>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} disabled={running}>
            <CloseOutlined />
          </button>
        </div>

        <div className="ws-modal-body" style={{ maxHeight: "78vh", overflowY: "auto" }}>
          <p style={{ color: "#8b949e", fontSize: 13, margin: "0 0 12px 0" }}>Paste the query list or upload files from <b style={{ color: "#58a6ff" }}>THUNGHIEM-bo-de-thi</b>. The app searches top keyframes for every query and packs one BTC-ready <b>submission.zip</b>.</p>

          <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
            <button 
              type="button" 
              className="ws-btn" 
              style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}
              onClick={() => fileInputRef.current?.click()}
              disabled={running}
            >
              <UploadOutlined /> Upload query files (.txt)
            </button>
            <input 
              ref={fileInputRef} 
              type="file" 
              multiple 
              accept=".txt,.csv" 
              hidden 
              onChange={handleFileUpload} 
            />
          </div>

          <div className="ws-field">
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <label className="ws-field-label">Query content</label>
              <span style={{ fontSize: 12, color: parsedQueries.length ? "#3fb950" : "#8b949e" }}>
                {parsedQueries.length > 0 ? `${parsedQueries.length} queries detected` : "No queries"}
              </span>
            </div>
            <textarea
              style={{
                width: "100%",
                height: 160,
                background: "#0d1117",
                color: "#c9d1d9",
                border: "1px solid #30363d",
                borderRadius: 6,
                padding: "8px 12px",
                fontSize: 12,
                fontFamily: "monospace",
                resize: "vertical"
              }}
              placeholder={`Example:\nquery-p1-1-kis: private spacecraft launch intro...\nquery-p1-2-kis: ambulance passing an intersection...\n...`}
              value={inputText}
              onChange={(e) => handleParseText(e.target.value)}
              disabled={running}
            />
          </div>

          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <div className="ws-field" style={{ flex: 1 }}>
              <label className="ws-field-label">Top K frames per query</label>
              <input 
                type="number" 
                min={1} 
                max={200} 
                value={topk} 
                onChange={(e) => setTopk(Number(e.target.value) || 100)} 
                disabled={running}
              />
            </div>
            <div className="ws-field" style={{ flex: 2 }}>
              <label className="ws-field-label">Output zip filename</label>
              <input 
                value={zipName} 
                onChange={(e) => setZipName(e.target.value)} 
                placeholder="submission.zip" 
                disabled={running}
              />
            </div>
          </div>

          {running && (
            <div style={{ background: "#161b22", padding: 12, borderRadius: 6, marginBottom: 16, border: "1px solid #30363d" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, color: "#58a6ff" }}>
                <LoadingOutlined spin /> <b>{progress.status}</b>
              </div>
              <div style={{ background: "#21262d", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div 
                  style={{ 
                    background: "#238636", 
                    height: "100%", 
                    width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                    transition: "width 0.3s ease"
                  }} 
                />
              </div>
            </div>
          )}

          {resultsSummary && (
            <div style={{ background: "#0d1117", padding: 10, borderRadius: 6, marginBottom: 16, border: "1px solid #238636" }}>
              <div style={{ color: "#3fb950", fontWeight: "bold", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                <CheckCircleOutlined /> Finished {resultsSummary.length} queries and exported {zipName}.
              </div>
              <div style={{ maxHeight: 120, overflowY: "auto", fontSize: 11, color: "#8b949e" }}>
                {resultsSummary.map((s, idx) => (
                  <div key={idx} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", borderBottom: "1px solid #21262d" }}>
                    <span><b>{s.name}</b> ({s.type.toUpperCase()}): {s.count} frames</span>
                    <span style={{ color: "#58a6ff" }}>{s.preview}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="ws-runbar">
            <button 
              className="ws-btn primary" 
              style={{ background: "#238636", borderColor: "#2ea043" }}
              onClick={handleRunBatch} 
              disabled={running || !parsedQueries.length}
            >
              {running ? <><LoadingOutlined spin /> Searching...</> : <><DownloadOutlined /> Run All & Export ZIP ({parsedQueries.length} Queries)</>}
            </button>
            <button className="ws-btn" onClick={onClose} disabled={running}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
