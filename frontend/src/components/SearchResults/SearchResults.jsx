import React from "react";
import ListCard from "./Card/ListCard";


const SearchResults = ({ searchType, require, setResult, result }) => {
  return (
    <React.Fragment>
      <div>
        <ListCard searchType={searchType} require={require} setResult={setResult} result={result}/>
      </div>
    </React.Fragment>
  );
};

export default SearchResults;
