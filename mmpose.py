import cv2
import numpy as np
import pandas as pd
import os
import torch
import multiprocessing
from mmpose.apis import inference_topdown, init_model

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

HAND_ID = [
    "wrist",
    "thumbCMC",
    "thumbMCP",
    "thumbIP",
    "thumbTip",
    "indexMCP",
    "indexPIP",
    "indexDIP",
    "indexTip",
    "middleMCP",
    "middlePIP",
    "middleDIP",
    "middleTip",
    "ringMCP",
    "ringPIP",
    "ringDIP",
    "ringTip",
    "littleMCP",
    "littlePIP",
    "littleDIP",
    "littleTip"
]
BODY_ID = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist"
]
BODY_DIC = {0:0, 5:2, 2:3, 8:4, 7:5, 12:6, 11:7, 14:8, 13:9, 16:10, 15:11}

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
def get_pose(file, model, lab, lab2id):

    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_duration = 1000 / fps
    current_timestamp = 0

    landmakr_dic = make_dic_arr()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        pose_results = inference_topdown(model, frame)
        
        # make inferance here


        
    return landmakr_dic

def make_dic_arr():
    dic = {}
    for part in HAND_ID:
        dic[part+"_right"+"_X"] = []
        dic[part+"_right"+"_Y"] = []
        dic[part+"_left"+"_X"] = []
        dic[part+"_left"+"_Y"] = []
    for part in BODY_ID:
        dic[part+"_X"] = []
        dic[part+"_Y"] = []
    dic["labels"] = []
    dic["video_fps"] = []
    dic["video_size_width"] = []
    dic["video_size_height"] = []
    dic["root_X"] = []
    dic["root_Y"] = []

    return dic


def main():
    train = './data/train/'
    val = './data/val/'
    test = './data/test/'
    
    train_lab = pd.read_csv("./data/labels/train_labels.csv", dtype=str)
    train_lab = train_lab.sort_values(train_lab.columns[0])
    val_lab = pd.read_csv("./data/labels/val_labels.csv", dtype=str)
    val_lab = val_lab.sort_values(val_lab.columns[0])
    test_lab = pd.read_csv("./data/labels/test_labels.csv", dtype=str)
    test_lab = test_lab.sort_values(test_lab.columns[0])
    all_lab = pd.concat([train_lab, test_lab, val_lab], ignore_index=True)
    lab2id = {lab:i for i, lab in enumerate(all_lab[all_lab.columns[1]].unique())}

    # Create empty CSV files with headers
    headers = make_dic_arr().keys()
    # pd.DataFrame(columns=headers).to_csv("WLASL2000_train_t.csv", index=False)
    # pd.DataFrame(columns=headers).to_csv("WLASL2000_val_t.csv", index=False)
    # pd.DataFrame(columns=headers).to_csv("WLASL2000_test_t.csv", index=False)    
    device = device = torch.device("mps")
    model = init_model(
        "./mmpose/blob/master/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py", 
        "./hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth", 
        device=device)


    for i, row in train_lab.iterrows():
        video_path = os.path.join(train, row[train_lab.columns[0]] + ".mp4")
        label = row[train_lab.columns[1]]

        dic = get_pose(video_path, model, label, lab2id)

if __name__ == "__main__":
    main()