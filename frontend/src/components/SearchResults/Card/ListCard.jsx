import React, { useEffect, useState } from "react";
import SingleTextSearchCard from "./SingleTextSearchCard";
import QASearchCard from "./QASearchCard";
import ImageSearchCard from "./ImageSearchCard";
import { Flex } from "antd";
import { Pagination, Spin } from "antd";
import {
  singleTextSearch,
  QASearch,
  OcrAndOdSearch,
  imageSearch,
  temporalSearch,
} from "../../../services/userService";
import styles from "./ListCard.module.css";
import TemporalSearchCard from "./TemporalSearchCard";
import { DeleteOutlined } from "@ant-design/icons";

const PAGE_SIZE = 40;

const ListCard = ({ searchType, require, setResult }) => {
  const [loading, setLoading] = useState(false);
  const [current, setCurrent] = useState(1);
  const [fullData, setFullData] = useState([]);
  const [data, setData] = useState([]);
  const [totalImage, setTotalImage] = useState();

  const handleDelete = (img) => {
    const newData = fullData.filter((item) => item.id !== img.id);
    setFullData(newData);
    setResult(newData);
    setTotalImage(newData.length);
    setData(newData.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE));
  };


  const getSimilarImage = async (img) => {
    setLoading(true);
    try {
      const response = await imageSearch({
        image: null,
        topk: 40,
        faiss_index: img.faiss_id_clip,
        clip: true,
        clipv2: false,
      });

      if (response && response.items) {
        setResult(response.items);
        setTotalImage(response.total_items);
        setFullData(response.items);
        setData(
          response.items.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE)
        );
      }
    } catch (error) {
      console.error("Error fetching similar images:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    getData();
  }, [require]);

  useEffect(() => {
    setData([]);
  }, [searchType]);

  useEffect(() => {
    getPageImages(current);
  }, [current, fullData]);

  const setPageData = (page) => {
    setCurrent(page);
  };

  const getData = async () => {
    setLoading(true);
    let data;
    if (require && Object.keys(require).length > 0) {
      switch (searchType) {
        case "Single Text Search":
          data = await singleTextSearch({
            query: require.query,
            topk: require.topk,
            clip: require.clip,
            clipv2: require.clipv2,
          });
          break;

        case "Q&A Search":
          data = await QASearch({
            query: require.query,
            topk: require.topk,
            clip: require.clip,
            clipv2: require.clipv2,
          });
          break;

        case "OCR and OD Search":
          data = await OcrAndOdSearch({
            query: require.query,
            topk: require.topk,
            ocr: require.ocr,
            od: require.objectDetection,
            bbox: require.bbox,
          });
          break;

        case "Image Search":
          data = await imageSearch({
            image: require.image,
            topk: require.topk,
            faiss_index: require.faiss_index,
            clip: require.clip,
            clipv2: require.clipv2,
          });
          break;

        case "Temporal Search":
          data = await temporalSearch({
            query: require.query,
            topk: require.topk,
            cascaded: require.cascaded,
          });
          break;
      }
    }

    if (data) {
      setResult(data.items);
      setTotalImage(data.total_items);
      setFullData(data.items);
    }
    setLoading(false);
  };
  const getPageImages = (page) => {
    const pageData = fullData.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
    setData(pageData);
  };

  return (
    <React.Fragment>
      <div className={styles.pageContainer}>
        <div className={styles.imageContainer}>
          {loading ? (
            <Flex
              justify="center"
              align="center"
              style={{ width: "100%", minHeight: "200px" }}
            >
              <Spin size="large" tip="Loading..." />
            </Flex>
          ) : (
            <Flex wrap gap="large">
              {searchType === "Single Text Search" ||
              searchType === "OCR and OD Search"
                ? data.map((item) => (
                    <SingleTextSearchCard
                      key={item.id}
                      img={item}
                      onDelete={handleDelete}
                      onGetSimilarImage={getSimilarImage}
                    />
                  ))
                : searchType === "Q&A Search"
                ? data.map((item) => <QASearchCard key={item.id} img={item} />)
                : searchType === "Image Search"
                ? data.map((item) => (
                    <ImageSearchCard
                      key={item.id}
                      img={item}
                      onDelete={handleDelete}
                      onGetSimilarImage={getSimilarImage}
                    />
                  ))
                : searchType === "Temporal Search"
                ? data.map((item) => (
                    <div
                      key={item.id}
                      style={{ display: "flex", flexDirection: "row" }}
                    >
                      <div className={styles.iconWrapper}>
                        <DeleteOutlined onClick={() => handleDelete(item)} />
                      </div>
                      <TemporalSearchCard img={item.frames} />
                    </div>
                  ))
                : null}
            </Flex>
          )}
        </div>
        <div className={styles.paginationContainer}>
          <Pagination
            current={current}
            onChange={setPageData}
            total={totalImage}
            showSizeChanger={false}
            pageSize={PAGE_SIZE}
          />
        </div>
      </div>
    </React.Fragment>
  );
};

export default ListCard;
