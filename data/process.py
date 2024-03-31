import json
import os

if __name__ == "__main__":
    
    # open json
    f = open('./WLASL_v0.3.json', 'r')
    content = json.load(f)

    # make train, val, test, folders
    if not os.path.exists("./train"):
        os.makedirs('./train')
    if not os.path.exists("./val"):
        os.makedirs('./val')
    if not os.path.exists("./test"):
        os.makedirs('./test')
    if not os.path.exists("./labels"):
        os.makedirs('./labels')
    
    # create labels
    train_lab = open('./labels/train_labels.csv', 'w')
    val_lab = open('./labels/val_labels.csv', 'w')
    test_lab = open('./labels/test_labels.csv', 'w')
    
    # pasrse through json
    for entry in content:
        label = entry['gloss']

        for instance in entry['instances']:
            split = instance['split']
            video_name = "{}.mp4".format(instance['video_id'])

            if os.path.exists('./videos/' + video_name):
                os.rename('./videos/' + video_name, './{}/{}'.format(split, video_name))
                if split == 'train':
                    train_lab.write('{},{}\n'.format(video_name, label))
                elif split == 'val':
                    val_lab.write('{},{}\n'.format(video_name, label))
                elif split == 'test':
                    test_lab.write('{},{}\n'.format(video_name, label))
                else:
                    print("fuck")
                
    
    train_lab.close()
    val_lab.close()
    test_lab.close()
    f.close()