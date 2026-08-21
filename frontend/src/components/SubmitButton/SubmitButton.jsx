import React, { useState } from "react";
import { Button, Input, Modal, Select } from "antd";
import { buildSubmissionCsv, makeSingleFileSubmissionZip, sanitizeQueryFileName } from "../../shared/submissionExport";

const SEARCH_TYPE_TO_QUERY_TYPE = {
  "Single Text Search": "kis",
  "OCR and OD Search": "kis",
  "Image Search": "kis",
  "Q&A Search": "qa",
  "Temporal Search": "trake",
};

const normalizeLegacyItem = (item) => ({
  ...item,
  videoKey: item.videoKey ?? item.video_key ?? item.video_id,
  frameKey: item.frameKey ?? item.frame_key ?? item.frame_id,
  globalFrameId: item.globalFrameId ?? item.global_frame_id,
  submissionFrameId: item.submissionFrameId ?? item.submission_frame_id ?? item.frame_idx ?? item.frameKey ?? item.frame_key ?? item.frame_id ?? item.globalFrameId ?? item.global_frame_id,
});

const SubmitButton = ({ result, searchType }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [queryType, setQueryType] = useState("kis");
  const [csvName, setCsvName] = useState("query-1-kis.csv");
  const [zipName, setZipName] = useState("submission.zip");
  const [answer, setAnswer] = useState("");

  const showModal = () => {
    const nextType = SEARCH_TYPE_TO_QUERY_TYPE[searchType] || "kis";
    setQueryType(nextType);
    setCsvName(`query-1-${nextType}.csv`);
    setIsModalOpen(true);
  };

  const handleOk = () => {
    const items = (result || []).map(normalizeLegacyItem);
    if (items.length === 0) {
      alert("No data to download");
      return;
    }
    if (queryType === "qa" && !items.some((item) => item.answer) && !answer.trim()) {
      alert("QA submission needs an answer");
      return;
    }

    const csvFileName = sanitizeQueryFileName(csvName, queryType);
    const csv = buildSubmissionCsv(items, queryType, answer);
    const blob = makeSingleFileSubmissionZip(csvFileName, csv);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeZipName = (zipName.trim() || "submission.zip").replace(/[\\/:*?"<>|]+/g, "_");
    a.download = safeZipName.toLowerCase().endsWith(".zip") ? safeZipName : `${safeZipName}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setIsModalOpen(false);
  };

  return (
    <>
      <div style={{ padding: "10px 0", fontSize: "18px", fontWeight: "bold", color: "#fff" }}>
        Submit results
      </div>

      <Button size="large" block onClick={showModal}>
        Submit
      </Button>

      <Modal title="Submit Results" open={isModalOpen} onOk={handleOk} onCancel={() => setIsModalOpen(false)}>
        <Select
          value={queryType}
          onChange={(value) => {
            setQueryType(value);
            setCsvName((current) => (current || "query-1-kis.csv").replace(/-(kis|qa|trake)(\.csv)?$/i, `-${value}.csv`));
          }}
          options={[
            { value: "kis", label: "KIS" },
            { value: "qa", label: "Q&A" },
            { value: "trake", label: "TRAKE" },
          ]}
          style={{ width: "100%", marginBottom: 12 }}
        />
        <Input value={csvName} onChange={(e) => setCsvName(e.target.value)} placeholder={`query-1-${queryType}.csv`} style={{ marginBottom: 12 }} />
        <Input value={zipName} onChange={(e) => setZipName(e.target.value)} placeholder="submission.zip" style={{ marginBottom: 12 }} />
        {queryType === "qa" ? <Input value={answer} onChange={(e) => setAnswer(e.target.value.slice(0, 100))} placeholder="Fallback answer" /> : null}
      </Modal>
    </>
  );
};

export default SubmitButton;
