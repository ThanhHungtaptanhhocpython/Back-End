import React, { useState } from "react";
import { Button, Input, Modal } from "antd";

const SubmitButton = ({ result, searchType }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [fileName, setFileName] = useState("");

  const showModal = () => {
    if (searchType === "Single Text Search" || searchType === "OCR and OD Search") {
      setFileName("query_$_kis");
    } else if (searchType === "Q&A Search") {
      setFileName("query_$_qa");
    } else {
      setFileName(`results_${Date.now()}`);
    }
    setIsModalOpen(true);
  };

  const handleOk = () => {
    console.log("Submitted results: ", result);

    if (!result || result.length === 0) {
      alert("No data to download");
      return;
    }

    let rows = "";

    // Tạo dữ liệu CSV
    if (searchType === "Q&A Search") {
      rows = result
        .map((item) => `${item.folder_key}_${item.video_key},${item.frame_key},${item.answer}`)
        .join("\n");
    } else if (searchType === "Single Text Search" || searchType === "OCR and OD Search" || searchType === "Image Search") {
      rows = result
        .map((item) => `${item.folder_key},${item.global_frame_id},${item.frame_name}`)
        .join("\n");
    }

    // Tạo Blob cho CSV
    const blob = new Blob([rows], { type: "text/csv;charset=utf-8;" });

    // Tạo link download
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;

    // Nếu user nhập tên file thì dùng, nếu không thì dùng mặc định
    const finalFileName = fileName.trim() !== "" ? fileName.trim() : `results_${Date.now()}`;
    a.download = `${finalFileName}.csv`;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setIsModalOpen(false);
  };

  const handleCancel = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <div
        style={{
          padding: "10px 0",
          fontSize: "18px",
          fontWeight: "bold",
          color: "#fff",
        }}
      >
        Submit results
      </div>

      <Button size="large" block onClick={showModal}>
        Submit
      </Button>

      <Modal
        title="Submit Results"
        open={isModalOpen}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Input
          placeholder="Enter file name (optional)"
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
        />
      </Modal>
    </>
  );
};

export default SubmitButton;
