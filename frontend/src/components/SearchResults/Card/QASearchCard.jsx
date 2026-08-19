import React from "react";
import { Image, Button } from "antd";
import styles from "./QASearchCard.module.css";
import { PlayCircleOutlined } from "@ant-design/icons";

const QASearchCard = (props) => {
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
                <div className={styles.answerPreviewContainer}>{props.img.answer}</div>
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
          {props.img.folder_key && props.img.video_key && props.img.frame_key
            ? `${props.img.folder_key}_${props.img.video_key}_${props.img.frame_key}`
            : ""}
          <div className={styles.answerContainer}>{props.img.answer}</div>
        </div>
      </div>
    </React.Fragment>
  );
};

export default QASearchCard;
