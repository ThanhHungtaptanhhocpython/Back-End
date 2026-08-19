import os
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
import json
import pandas as pd
class VisualEncoding:

  def __init__(self,
    classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
              ),
    row_str = ["0", "1", "2", "3", "4", "5", "6"],
    col_str = ["a", "b", "c", "d", "e", "f", "g"]):

    self.classes = classes
    self.classes2idx = dict()
    for i, class_ in enumerate(classes):
        self.classes2idx[class_] = i
    self.n_row = len(row_str)
    self.n_col = len(col_str)

    x_pts = np.linspace(0, 1, self.n_row+1)
    y_pts = np.linspace(0, 1, self.n_col+1)

    self.grid_bboxes = []
    self.grid_labels = []
    for i in range(self.n_row):
        for j in range(self.n_col):
            label = col_str[j] + row_str[i]
            self.grid_bboxes.append([x_pts[j], y_pts[i], x_pts[j+1], y_pts[i+1]])
            self.grid_labels.append(label)

    self.grid_bboxes = np.array(self.grid_bboxes)

  def visualize_grid(self, grid_vis=None):
        if grid_vis is None:
            grid_vis = np.zeros((500, 500, 1))

        vis_h, vis_w, _ = grid_vis.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2
        for i in range(self.n_row*self.n_col):
            x_start, y_start, x_end, y_end = self.grid_bboxes[i]
            label = self.grid_labels[i]
            org = (int((x_start + (x_end-x_start)/2)*vis_w), int((y_start + (y_end-y_start)/2)*vis_h))

            # Draw text
            grid_vis = cv2.putText(grid_vis, label, org, font, fontScale, color, thickness, cv2.LINE_AA)
            # Draw grid
            grid_vis = cv2.rectangle(grid_vis, (int(x_start*vis_w), int(y_start*vis_h)), (int(x_end*vis_w), int(y_end*vis_h)), color, thickness)
        plt.imshow(grid_vis)

  def encode_bboxes(self, bboxes, labels):
        '''
        Args:
            bboxes: np.array: (n_bboxes, 4) - expected normalized bbox in form (x0, y0, x1, y1)
            labels: np.array: (n_bboxes, )
        '''
        iou = box_iou(torch.as_tensor(bboxes), torch.as_tensor(self.grid_bboxes))
        bboxes_idx, locs_idx = np.nonzero(iou.numpy())

        context = []
        for bbox_idx, loc_idx in zip(bboxes_idx, locs_idx):
            context.append(self.grid_labels[loc_idx] + self.classes[labels[bbox_idx]].replace(" ", ""))
        context = ' '.join(map(str, context))
        return context

  def encode_classes(self, labels):
        '''
        Args:
            labels: np.array: (n_bboxes, )
        '''
        unique_classes, counts = np.unique(labels, return_counts=True)
        context = []
        for unique_class, count in zip(unique_classes, counts):
            for i in range(count):
                context.append(self.classes[unique_class].replace(" ", "") + str(i))
        context = ' '.join(map(str, context))
        return context

  def encode_numbers(self, labels):
        '''
        Args:
            labels: np.array: (n_bboxes, )
        '''
        unique_classes, counts = np.unique(labels, return_counts=True)
        context = []
        for unique_class, count in zip(unique_classes, counts):
            context.append(self.classes[unique_class].replace(" ", "") + str(count))
        context = ' '.join(map(str, context))
        return context

  def encode(self, bboxes=None, labels=None, bboxes_colors=None, colors=None):
        '''
        Args:
            bboxes: np.array: (n_bboxes, 4) - expected normalized bbox in form (x0, y0, x1, y1)
            labels: np.array: (n_bboxes, )
        '''
        results = dict()
        if bboxes is not None:
            results['bbox'] = self.encode_bboxes(bboxes, labels)
            results['class'] = self.encode_classes(labels)
        else:
            results['bbox'] = results['class'] = None


        return results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def download_dataset():
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("trietdeptrai/frames2go")
    return path

def prepocess_data(keyframes_dir):
    all_keyframe_paths = dict()
    for part in sorted(os.listdir(keyframes_dir)):
        data_part = part # L01, L02 for ex
        all_keyframe_paths[data_part] =  dict()

    for data_part in sorted(all_keyframe_paths.keys()):
        data_part_path = f'{keyframes_dir}/{data_part}'
        video_dirs = sorted(os.listdir(data_part_path))
        video_ids = [video_dir.split('_')[-1] for video_dir in video_dirs]
        for video_id, video_dir in zip(video_ids, video_dirs):
            keyframe_paths = sorted(glob.glob(f'{data_part_path}/{video_dir}/*.webp'))
            all_keyframe_paths[data_part][video_id] = keyframe_paths
            
    return all_keyframe_paths

def get_OD_model():
    model = YOLO("yolo12x.pt")
    return model
def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--batch_size", type=int,default=256, help="Batch size")
    args = parser.parse_args()
    return args 
import argparse

if __name__ == "__main__":
    args = setup_args()
    bs = args.batch_size
    keyframes_dir = download_dataset()
    print(keyframes_dir)
    all_keyframe_paths = prepocess_data(keyframes_dir)
    
    encoder = VisualEncoding()
    for key, video_keyframe_paths in tqdm(all_keyframe_paths.items()):
        os.makedirs(f"./context_encoded/OD/{key}", exist_ok=True)
        video_ids = sorted(video_keyframe_paths.keys())
        model = get_OD_model()
        for video_id in tqdm(video_ids):
            save_path = f"./context_encoded/OD/{key}/{video_id}.json"

            if os.path.exists(save_path):
                print(f"[Skip] {save_path} đã có, bỏ qua...")
                continue

            image_paths = video_keyframe_paths[video_id]
            video_result_dict = {}
            csv_path = f"{keyframes_dir}/{key}/{video_id}/{video_id}_keyframes_metadata.csv"
            df = pd.read_csv(csv_path)
            frameid_map = dict(zip(df["keyframe_filename"], df["global_frame_index"]))

            for i in range(0, len(image_paths), bs):
                batch_paths = image_paths[i:i+bs]
                results = model(batch_paths, conf=0.5, device=device, verbose=False)

                for img_path, result in zip(batch_paths, results):
                    frame_name = os.path.basename(img_path)
                    bboxes = result.boxes.xyxyn.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)

                    image = cv2.imread(img_path)
                    H, W = image.shape[:2]

                    if len(bboxes) == 0:
                        video_result_dict[frame_name] = {
                            "bbox": "",
                            "class": "",
                            "number": "",
                            "global_frame_id": int(frameid_map[frame_name])
                        }
                        continue

                    encoded_bbox = encoder.encode_bboxes(bboxes, labels)
                    encoded_class = encoder.encode_classes(labels)
                    encoded_number = encoder.encode_numbers(labels)

                    video_result_dict[frame_name] = {
                        "bbox": encoded_bbox,
                        "class": encoded_class,
                        "number": encoded_number,
                        "global_frame_id": int(frameid_map[frame_name])
                    }

                    del image
                del results
                torch.cuda.empty_cache()

            with open(save_path, "w") as f:
                json.dump(video_result_dict, f, indent=2)

            del video_result_dict
