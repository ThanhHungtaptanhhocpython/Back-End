import React, { useState } from "react";
import { TreeSelect } from "antd";
import SingleTextSearch from "./SearchContent/SingleTextSearch";
import TemporalSearch from "./SearchContent/TemporalSearch";
import QASearch from "./SearchContent/QASearch";
// import OcrAndOdSearch from "./SearchContent/OcrAndOdSearch";
// import ImageSearch from "./SearchContent/ImageSearch";
import styles from "./DropMenu.module.css";
import OcrAndOdSearch from "./SearchContent/OcrAndOdSearch";
import ImageSearch from "./SearchContent/ImageSearch";

const treeData = [
  {
    title: "Text Search",
    children: [
      {
        title: "Single Text Search",
        value: "Single Text Search",
      },
      {
        title: "OCR and OD Search",
        value: "OCR and OD Search",
      },
    ],
  },
  {
    title: "Image Search",
    value: "Image Search",
  },
  {
    title: "Temporal Search",
    value: "Temporal Search",
  },
];

const DropMenu = ({ setSearchType, setRequire }) => {
  const [value, setValue] = useState("Single Text Search");
  const onChange = (newValue) => {
    setValue(newValue);
    setSearchType(newValue);
    setRequire({});
  };

  const renderSearchComponents = () => {
    switch (value) {
      case "Single Text Search":
        return <SingleTextSearch setRequire={setRequire} />;
      case "OCR and OD Search":
        return <OcrAndOdSearch setRequire={setRequire}/>;
      case "Image Search":
        return <ImageSearch setRequire={setRequire} />;
      case "Temporal Search":
        return <TemporalSearch setRequire={setRequire} />;
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.dropMenu}>
        <TreeSelect
          style={{ width: "100%" }}
          size="large"
          value={value}
          styles={{
            popup: { root: { maxHeight: 500, overflow: "auto" } },
          }}
          treeData={treeData}
          placeholder="Please select"
          treeDefaultExpandAll
          onChange={onChange}
        />
      </div>
      <div className={styles.searchBox}>{renderSearchComponents()}</div>
    </div>
  );
};

export default DropMenu;
