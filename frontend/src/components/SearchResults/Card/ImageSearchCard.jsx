import React from "react";
import { Image, Button } from "antd";
import styles from "./SingleTextSearchCard.module.css";
import { PlayCircleOutlined } from "@ant-design/icons";
import { DeleteOutlined, PlusSquareOutlined } from "@ant-design/icons";

const ImageSearchCard = (props) => {
  const onDelete = () => {
    props.onDelete(props.img);
  };
  const onGetSimilarImage = () => {
    props.onGetSimilarImage(props.img);
  };
  return (
    <React.Fragment>
      <div className={styles.imageContainer}>
        <Image
          preview={{
            destroyOnHidden: true,
            imageRender: () => (
              <div className={styles.previewContainer}>
                <img
                  style={{
                    borderRadius: "8px",
                  }}
                  src={`data:image/webp;base64,${props.img.image}`}
                  alt={
                    props.img.folder_key &&
                    props.img.video_key &&
                    props.img.frame_key
                      ? `${props.img.folder_key}_${props.img.video_key}_${props.img.frame_key}`
                      : ""
                  }
                />
                <Button
                  className={styles.playBtn}
                  icon={<PlayCircleOutlined />}
                  href={`${props.img.link}&=${props.img.timestamp}`}
                  target="_blank"
                >
                  Play
                </Button>
              </div>
            ),
            toolbarRender: () => null,
          }}
          style={{
            padding: "0.5rem",
          }}
          width={200}
          src={`data:image/webp;base64,${props.img.image}`}
        />
        <div className={styles.imageTitle}>
          {props.img.folder_key && props.img.global_frame_id && props.img.frame_name
            ? `${props.img.folder_key}_${props.img.frame_name}_${props.img.global_frame_id}`
            : ""}
        </div>
        <div className={styles.iconWrapper}>
          <DeleteOutlined onClick={onDelete} />
          <PlusSquareOutlined onClick={onGetSimilarImage} />
        </div>
      </div>
    </React.Fragment>
  );
};

export default ImageSearchCard;
