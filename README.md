# cv-tennis-analysis
 
## Detector engine
We used Ultralytics **YoloV5** to detect the players, raquets and the tennis ball. However, the model was not able to track the tenning ball properly and hence we had to fine tune it.<br>
For retraining, we used a pre-annotated dataset from [Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6#)<br>
Since this dataset only has annotations for tennis balls, we has to merge the outputs of yolov8 for player and raquet detection, and yolov5 for tennis ball detection.<br>