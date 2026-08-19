import React, { useState } from "react";
import { Select } from "antd";
const isDisabled = true;
const HistoryButton = () => {
  const [searchHistory, setSearchHistory] = useState(["ABC", "XYZ", "MNPQ"]);
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
        History
      </div>

      <Select
        size="large"
        showSearch
        placeholder="Tìm kiếm hoặc chọn từ lịch sử"
        optionFilterProp="children"
        style={{
          width: "100%",
          opacity: isDisabled ? 0.5 : 1,
          pointerEvents: isDisabled ? "none" : "auto",
        }}
        onSearch={(value) => {
          if (value && !searchHistory.includes(value)) {
            setSearchHistory((prev) => [value, ...prev]);
          }
        }}
      >
        <Select.OptGroup label="Search History">
          {searchHistory.map((item, index) => (
            <Select.Option key={index} value={item}>
              {item}
            </Select.Option>
          ))}
        </Select.OptGroup>
      </Select>
    </>
  );
};

export default HistoryButton;
