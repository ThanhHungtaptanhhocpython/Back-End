import React from "react";
import { Image, Button } from "antd";
import styles from "./TemporalSearchCard.module.css";
import { PlayCircleOutlined } from "@ant-design/icons";
import { DeleteOutlined } from "@ant-design/icons";

const TemporalSearchCard = (props) => {
  return (
    <div className={styles.imageItem}>
      {props.img.map((item) => (
        <div key={item.id} >
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
                      src={`data:image/webp;base64,${item.image}`}
                      alt={`${item.folder_key}_${item.video_key}_${item.frame_key}`}
                    />
                    <Button
                      className={styles.playBtn}
                      icon={<PlayCircleOutlined />}
                      href={`${item.link}&=${item.timestamp}`}
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
              src={`data:image/webp;base64,${item.image}`}
            />
            <div className={styles.imageTitle}>
              {item.folder_key && item.video_key && item.frame_key
                ? `${item.folder_key}_${item.video_key}_${item.frame_key}`
                : ""}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default TemporalSearchCard;
