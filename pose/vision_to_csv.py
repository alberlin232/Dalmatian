import pandas as pd
import os

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


CONV_DIC = {
    "wrist":"wrist",
    "thumbCMC":"thumbCMC",
    "thumbMCP":"thumbMCP",
    "thumbIP":"thumbIP",
    "thumbTip":"thumbTip",
    "indexFingerMCP":"indexMCP",
    "indexFingerPIP":"indexPIP",
    "indexFingerDIP":"indexDIP",
    "indexFingerTip":"indexTip",
    "middleFingerMCP":"middleMCP",
    "middleFingerPIP":"middlePIP",
    "middleFingerDIP":"middleDIP",
    "middleFingerTip":"middleTip",
    "ringFingerMCP":"ringMCP",
    "ringFingerPIP":"ringPIP",
    "ringFingerDIP":"ringDIP",
    "ringFingerTip":"ringTip",
    "pinkyMCP":"littleMCP",
    "pinkyPIP":"littlePIP",
    "pinkyDIP":"littleDIP",
    "pinkyTip":"littleTip",
    "nose": "nose",
    "neck": "neck",
    "rightEye": "rightEye",
    "leftEye": "leftEye",
    "rightEar": "rightEar",
    "leftEar": "leftEar",
    "rightShoulder": "rightShoulder",
    "leftShoulder": "leftShoulder",
    "rightElbow": "rightElbow",
    "leftElbow": "leftElbow",
    "rightWrist":"rightWrist",
    "leftWrist": "leftWrist"
}

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

def drop_duplicate_columns(df):
    # Strip suffix and check duplicates
    stripped_cols = df.columns.str.replace(r'\.\d+$', '', regex=True)
    duplicates = stripped_cols.duplicated(keep='first')
    return df.loc[:, ~duplicates]

if __name__ == "__main__":
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
    
    headers = make_dic_arr()

    dir_path = "./data/test"
    for i, row in test_lab.iterrows():
        csv_path = os.path.join(test, row[test_lab.columns[0]] + ".csv")
        label = row[test_lab.columns[1]]
        df = pd.read_csv(csv_path)
        df = drop_duplicate_columns(df)
        dic = df.to_dict(orient="list")
        length = len(dic["wrist_right_X"])
        for key in dic.keys():
            keys = key.split("_")
            if len(keys) == 3:
                new_key = CONV_DIC[keys[0]] + "_" + keys[1] + "_" + keys[2]
            elif len(keys) == 2:
                new_key = CONV_DIC[keys[0]] + "_" + keys[1].split(".")[0]
            headers[new_key].append(dic[key])
        headers["labels"].append(lab2id[label])
        headers["video_size_width"].append(0)
        headers["video_size_height"].append(0)
        headers["video_fps"].append(30)
        headers["root_Y"].append([0]*length)
        headers["root_X"].append([0]*length)
    pd.DataFrame().from_dict(headers).to_csv("./WLASL2000_test_vision.csv", index=False)
    
