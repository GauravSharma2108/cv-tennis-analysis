# (IN PROGRESS) cv-tennis-analysis
 
## Detector engine
Ultralytics **YoloV5** was used to detect the players, raquets and the tennis ball. However, the model was not able to track the tenning ball properly and hence it was fine tuned.<br>
For retraining, a pre-annotated dataset was used from [Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6#)<br>
Since this dataset only had annotations for tennis balls, the outputs of yolov8 and yolov5 were merged for player detection and tennis ball detection respectively.<br>

Additionally, another model was trained to detect the keypoints of the court. The dataset to train this model was sourced from [this GitHub repo from yastrebksv](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file).
