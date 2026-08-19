import React from "react";
import DropMenu from "./DropMenu/DropMenu";
import styles from "./SearchHeader.module.css";

const SearchHeader = ({ setSearchType, setRequire }) => {
  
  return (
    <header className={styles.headerContainer}>
      <div>
        <DropMenu setSearchType={setSearchType} setRequire={setRequire} />
      </div>
    </header>
  );
};

export default SearchHeader;
