import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image_hand(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_landmarks_on_image_pose(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
def get_pose(file):
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
            frame_duration = 1000 / fps
            current_timestamp = 0
            arr = []
            while cap.isOpened():
                ret, frame = cap.read()
                # frame = cv2.resize(frame, (224, 224))
                if not ret:
                    break
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_landmarker_result = landmarker_hand.detect_for_video(mp_image, int(current_timestamp))
                pose_landmerker_result = landmarker_pose.detect_for_video(mp_image, int(current_timestamp))
                current_timestamp += frame_duration    
            
                hand_marks = hand_landmarker_result.hand_landmarks
                handiness = hand_landmarker_result.handedness

                pose_marks = pose_landmerker_result.pose_landmarks
                
                content = np.zeros(154)
                # 0-20: left-x
                # 21-42: left-y
                # 43-64: right-x
                # 65-86: right-y
                # 87-120: pose-x
                # 121-154: pose-y
                for idx in range(len(hand_marks)):
                    if handiness[idx][0].category_name == "Left":
                        for i in range(len(hand_marks[idx])):
                            content[i] = hand_marks[idx][i].x
                            content[i+21] = hand_marks[idx][i].y
                    elif handiness[idx][0].category_name == "Right":
                        for i in range(len(hand_marks[idx])):
                            content[i+43] = hand_marks[idx][i].x
                            content[i+65] = hand_marks[idx][i].x

                for idx in range(len(pose_marks)):
                    for i in range(len(pose_marks[idx])):
                        content[i+87] = pose_marks[idx][i].x
                        content[i+121] = pose_marks[idx][i].y
                annotated_image = draw_landmarks_on_image_hand(mp_image.numpy_view(), hand_landmarker_result)
                annotated_image = draw_landmarks_on_image_pose(annotated_image, pose_landmerker_result)

                cv2.imshow('Hand Landmarks', annotated_image)
                cv2.imwrite('example_{}.jpg'.format(current_timestamp),annotated_image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                arr.append(content)
        cap.release()
        cv2.destroyAllWindows()
    return np.array(arr)
if __name__ == "__main__":
    train = './data/train/'
    val = './data/val/'
    test = './data/test/'
            
    for vid in os.listdir(train):
        if not os.path.exists(train+vid[:-3]+"txt"):
            res = get_pose(train+vid)
            print(res)
            np.savetxt(train+vid[:-3]+"txt", res, delimiter=',')
    for vid in os.listdir(val):
        if not os.path.exists(val+vid[:-3]+"txt"):
            res = get_pose(val+vid)
            print(res)
            np.savetxt(val+vid[:-3]+"txt", res, delimiter=',')

    for vid in os.listdir(test):
        if not os.path.exists(test+vid[:-3]+"txt"):
            res = get_pose(test+vid)
            print(res)
            np.savetxt(test+vid[:-3]+"txt", res, delimiter=',')