import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import os
import multiprocessing

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
def get_pose(file, lab):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options_hand = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2)
    options_pose = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./pose_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)

    with HandLandmarker.create_from_options(options_hand) as landmarker_hand:
        with PoseLandmarker.create_from_options(options_pose) as landmarker_pose:
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_landmarker_result = landmarker_hand.detect_for_video(mp_image, int(current_timestamp))
                pose_landmerker_result = landmarker_pose.detect_for_video(mp_image, int(current_timestamp))
                current_timestamp += frame_duration    
            
                hand_marks = hand_landmarker_result.hand_landmarks
                handiness = hand_landmarker_result.handedness

                pose_marks = pose_landmerker_result.pose_landmarks
                
                # 0-20: left-x
                # 21-42: left-y
                # 43-64: right-x
                # 65-86: right-y
                # 87-120: pose-x
                # 121-154: pose-y
                left = False
                right = False
                for idx in range(len(hand_marks)):
                    if handiness[idx][0].category_name == "Left" and not left:
                        left = True
                        for i in range(len(hand_marks[idx])):
                            landmakr_dic[HAND_ID[i]+"_left"+"_X"].append(hand_marks[idx][i].x)
                            landmakr_dic[HAND_ID[i]+"_left"+"_Y"].append(hand_marks[idx][i].y)
                    if handiness[idx][0].category_name == "Right" and not right:
                        right = True
                        for i in range(len(hand_marks[idx])):
                            landmakr_dic[HAND_ID[i]+"_right"+"_X"].append(hand_marks[idx][i].x)
                            landmakr_dic[HAND_ID[i]+"_right"+"_Y"].append(hand_marks[idx][i].y)
                if not left:
                    for i in range(len(HAND_ID)):
                        landmakr_dic[HAND_ID[i]+"_left"+"_X"].append(0)
                        landmakr_dic[HAND_ID[i]+"_left"+"_Y"].append(0)
                if not right:
                    for i in range(len(HAND_ID)):
                        landmakr_dic[HAND_ID[i]+"_right"+"_X"].append(0)
                        landmakr_dic[HAND_ID[i]+"_right"+"_Y"].append(0)

                # Body Time!!
                body = False
                for idx in range(len(pose_marks)):
                    body = True
                    for i in range(len(pose_marks[idx])):
                        if i in BODY_DIC:
                            landmakr_dic[BODY_ID[BODY_DIC[i]]+"_X"].append(pose_marks[idx][i].x)
                            landmakr_dic[BODY_ID[BODY_DIC[i]]+"_Y"].append(pose_marks[idx][i].y)
                    landmakr_dic["neck_X"].append((landmakr_dic["rightShoulder_X"] + landmakr_dic["leftShoulder_X"])/2)
                    landmakr_dic["neck_Y"].append((landmakr_dic["rightShoulder_Y"] + landmakr_dic["leftShoulder_Y"])/2)
                    landmakr_dic["root_X"].append(0)
                    landmakr_dic["root_Y"].append(0)
              
                if not body:
                    for i in range(len(BODY_ID)):
                        landmakr_dic[BODY_ID[i]+"_X"].append(0)
                        landmakr_dic[BODY_ID[i]+"_Y"].append(0)
                    # landmakr_dic["neck_X"].append(0)
                    # landmakr_dic["neck_Y"].append(0)
                    landmakr_dic["root_X"].append(0)
                    landmakr_dic["root_Y"].append(0)   

            landmakr_dic["video_fps"] = round(fps)
            landmakr_dic["video_size_width"] = width
            landmakr_dic["video_size_height"] = height
            landmakr_dic["labels"] = lab
                # annotated_image = draw_landmarks_on_image_hand(mp_image.numpy_view(), hand_landmarker_result)
                # annotated_image = draw_landmarks_on_image_pose(annotated_image, pose_landmerker_result)

                # cv2.imshow('Hand Landmarks', annotated_image)
                # cv2.imwrite('example_{}.jpg'.format(current_timestamp),annotated_image)
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
        cap.release()
        cv2.destroyAllWindows()
    for key in landmakr_dic.keys():
        landmakr_dic[key] = [landmakr_dic[key]]
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

    # Create empty CSV files with headers
    headers = make_dic_arr().keys()
    pd.DataFrame(columns=headers).to_csv("64ALS.csv", index=False)
    
    videos = os.listdir('./data/ISA64')
    
    # Use a multiprocessing pool to process videos in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    res = []
    for video in videos:
        video_path = os.path.join('./data/ISA64', video)
        print(video_path)
        label = video[-15:-12]
        dic = pool.apply_async(get_pose, args=(video_path, int(label)))
        res.append(dic)
    pool.close()
    for f_res in res:
        result_dic = f_res.get(timeout=60)
        pd.DataFrame.from_dict(result_dic).to_csv("64ALS.csv", mode='a', index=False, header=False)

    pool.join()

if __name__ == "__main__":
    main()