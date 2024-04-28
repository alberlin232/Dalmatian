# Dalmatian

### Download Data

To download the WLASL data, in the data folder run the download_data.sh file with bash. This will download all data into a video folder. Then run the process.py script to seperate them into train, validation, and testing sets.


### Extracting Poses

To extract the WLASL poses with Mediapipe you must first go install Mediapipe as specified on their website. To extract simply run the pose.py file in the pose directory from the root folder.

To extract with Vision API, firstly you must need a deviec that can run MacOS 13 or later. You can simply run the main.swift file and then the vision to csv.py file after.

### SPOTER

To run spoter you must cd into the spoter_import dir and run python -m train --set_training_set "training set path" --set_validation_set "validation set path" --class_num x. You also have to change line 37 of the utils.py file in the model folder to proper use the validation method. There are other setting that can be found in the train.py file.

