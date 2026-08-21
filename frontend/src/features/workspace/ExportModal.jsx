import { useRef, useState } from "react";
import { CloseOutlined, ExportOutlined } from "@ant-design/icons";
import useDialogFocus from "../../hooks/useDialogFocus";
import { buildSubmissionCsv, makeSubmissionZip, sanitizeQueryFileName, queryTypeFromSearchType } from "../../shared/submissionExport";

export default function ExportModal({ open, items, activeTabResults = [], tabs = [], onClose, toast }) {
  const [queryType, setQueryType] = useState("kis");
  const [source, setSource] = useState("search"); // 'search' | 'tray' | 'allTabs'
  const [csvName, setCsvName] = useState("query-1-kis.csv");
  const [zipName, setZipName] = useState("submission.zip");
  const [answer, setAnswer] = useState("");
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);
  if (!open) return null;

  const maxFramesPerCsv = 100;
  
  // Choose items based on source
  let exportSourceItems = [];
  if (source === "tray") {
    exportSourceItems = items;
  } else if (source === "search") {
    exportSourceItems = (activeTabResults || []).slice(0, maxFramesPerCsv);
  } else if (source === "allTabs") {
    // Collect top 100 for each tab as separate query files
    exportSourceItems = (tabs || []).flatMap((tab, idx) => {
      const qIndex = idx + 1;
      const qType = queryTypeFromSearchType(tab?.searchType);
      return (tab?.results || []).slice(0, maxFramesPerCsv).map((item) => ({
        ...item,
        __submission: {
          key: tab.key,
          queryIndex: qIndex,
          queryType: qType,
        },
      }));
    });
  }

  const hasSubmissionMeta = exportSourceItems.some((item) => item.__submission?.queryType);
  const effectiveCsvName = sanitizeQueryFileName(csvName, queryType);
  const orderedItems = exportSourceItems
    .map((item, order) => ({ item, order }))
    .sort((a, b) => {
      const aIndex = a.item.__submission?.queryIndex ?? Number.MAX_SAFE_INTEGER;
      const bIndex = b.item.__submission?.queryIndex ?? Number.MAX_SAFE_INTEGER;
      return aIndex - bIndex || a.order - b.order;
    })
    .map(({ item }) => item);

  const typedGroups = orderedItems.reduce((groups, item) => {
    const type = item.__submission?.queryType || queryType;
    const qIdx = item.__submission?.queryIndex;
    const groupKey = qIdx !== undefined ? `${type}_${qIdx}` : type;
    if (!groups.has(groupKey)) groups.set(groupKey, { type, qIdx, items: [] });
    groups.get(groupKey).items.push(item);
    return groups;
  }, new Map());

  const files = [];

  if (!hasSubmissionMeta && orderedItems.length <= maxFramesPerCsv) {
    files.push({ name: effectiveCsvName, queryType, content: buildSubmissionCsv(orderedItems, queryType, answer) });
  } else {
    Array.from(typedGroups.values()).forEach((group) => {
      const { type, qIdx, items: groupItems } = group;
      for (let start = 0; start < groupItems.length; start += maxFramesPerCsv) {
        const chunk = groupItems.slice(start, start + maxFramesPerCsv);
        const fileName = qIdx !== undefined 
          ? `query-p1-${qIdx}-${type}.csv` 
          : `query-${files.length + 1}-${type}.csv`;
        files.push({
          name: fileName,
          queryType: type,
          content: buildSubmissionCsv(chunk, type, answer),
        });
      }
    });
  }

  const previewRows = files.flatMap((file) => {
    const rows = file.content ? file.content.split("\n") : [];
    return [sanitizeQueryFileName(file.name, file.queryType), ...rows.slice(0, 5), ...(rows.length > 5 ? [`... +${rows.length - 5} more`] : [])];
  });
  const preview = previewRows.join("\n");
  const needsQaAnswer = files.some((file) => file.queryType === "qa");

  const download = () => {
    if (exportSourceItems.length === 0) {
      toast.warning("No frames to export. Run a search or keep items in tray.");
      return;
    }
    const hasQaWithoutAnswer = needsQaAnswer && !exportSourceItems.some((item) => item.answer) && !answer.trim();
    if (hasQaWithoutAnswer) {
      toast.warning("QA submission needs an answer");
      return;
    }

    const blob = makeSubmissionZip(files);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeZipName = (zipName.trim() || "submission.zip").replace(/[\\/:*?"<>|]+/g, "_");
    a.download = safeZipName.toLowerCase().endsWith(".zip") ? safeZipName : `${safeZipName}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${exportSourceItems.length} result(s) as a root-level BTC submission zip`);
    onClose();
  };

  const downloadCsv = () => {
    if (exportSourceItems.length === 0) {
      toast.warning("No frames to export. Run a search or keep items in tray.");
      return;
    }
    const hasQaWithoutAnswer = needsQaAnswer && !exportSourceItems.some((item) => item.answer) && !answer.trim();
    if (hasQaWithoutAnswer) {
      toast.warning("QA submission needs an answer");
      return;
    }

    files.forEach((file) => {
      const blob = new Blob([file.content || ""], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const safeName = sanitizeQueryFileName(file.name || effectiveCsvName, file.queryType || queryType);
      a.download = safeName;
      a.click();
      URL.revokeObjectURL(url);
    });

    toast.success(`Exported ${files.length} CSV file(s) directly`);
    onClose();
  };

  const selectQueryType = (type) => {
    setQueryType(type);
    setCsvName((current) => {
      const base = current || `query-1-${type}.csv`;
      return base.replace(/-(kis|qa|trake)(\.csv)?$/i, `-${type}.csv`);
    });
  };

  return (
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialogRef} className="ws-modal" role="dialog" aria-modal="true" aria-label="Export submission" tabIndex={-1}>
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <ExportOutlined /> Export Submission (BTC Standard Zip)
          </div>
          <span className="ws-dim" style={{ fontSize: "11px", letterSpacing: "1px" }}>
            {exportSourceItems.length} FRAMES Â· ROOT-LEVEL ZIP READY
          </span>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Close (Esc)">
            <CloseOutlined />
          </button>
        </div>
        <div className="ws-modal-body">
          <div className="ws-field">
            <label className="ws-field-label">Export Source</label>
            <div className="ws-exp-fmt" style={{ marginBottom: 8 }}>
              <button
                type="button"
                className={`ws-type ${source === "search" ? "active" : ""}`}
                style={{ flex: 1 }}
                onClick={() => setSource("search")}
              >
                Top 100 Search Results ({Math.min(100, activeTabResults.length)})
              </button>
              <button
                type="button"
                className={`ws-type ${source === "tray" ? "active" : ""}`}
                style={{ flex: 1 }}
                onClick={() => setSource("tray")}
              >
                Selection Tray ({items.length})
              </button>
              <button
                type="button"
                className={`ws-type ${source === "allTabs" ? "active" : ""}`}
                style={{ flex: 1 }}
                onClick={() => setSource("allTabs")}
              >
                All Query Tabs ({tabs.length} queries)
              </button>
            </div>
          </div>

          <div className="ws-field">
            <label className="ws-field-label">Query Type</label>
            <div className="ws-exp-fmt">
              {["kis", "qa", "trake"].map((type) => (
                <button
                  key={type}
                  className={`ws-type ${queryType === type ? "active" : ""}`}
                  style={{ minWidth: 100 }}
                  onClick={() => selectQueryType(type)}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="ws-field">
            <label className="ws-field-label">CSV filename (Root Level)</label>
            <input value={csvName} onChange={(e) => setCsvName(e.target.value)} placeholder={`query-1-${queryType}.csv`} />
          </div>
          <div className="ws-field">
            <label className="ws-field-label">Zip filename</label>
            <input value={zipName} onChange={(e) => setZipName(e.target.value)} placeholder="submission.zip" />
          </div>
          {needsQaAnswer ? (
            <div className="ws-field">
              <label className="ws-field-label">Fallback answer</label>
              <input value={answer} onChange={(e) => setAnswer(e.target.value.slice(0, 100))} placeholder="Used when selected result has no answer" />
            </div>
          ) : null}
          <div className="ws-field">
            <label className="ws-field-label">Preview (File Structure in Zip)</label>
            <div className="ws-exp-preview">{preview}</div>
          </div>
          <div className="ws-runbar" style={{ display: "flex", gap: "8px" }}>
            <button className="ws-btn primary" onClick={downloadCsv} disabled={exportSourceItems.length === 0} style={{ flex: 1 }}>
              <ExportOutlined /> Download CSV ({files.length} file{files.length > 1 ? "s" : ""})
            </button>
            <button className="ws-btn" onClick={download} disabled={exportSourceItems.length === 0} style={{ flex: 1 }}>
              <ExportOutlined /> Download ZIP
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
