# (IN PROGRESS) cv-tennis-analysis
 
## Detector engine
Ultralytics **YoloV8** was used to detect the players, and **YoloV5** to detect the tennis ball. However, the model was not able to track the tenning ball properly and hence it was fine tuned.<br>
For retraining, a pre-annotated dataset was used from [Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6#).Since this dataset only had annotations for tennis balls, the outputs of yolov8 and yolov5 were merged for player detection and tennis ball detection respectively.<br>

To detect the keypoints on the court, a pretrained ResNet50 model was used, which was fined-tuned by adding a FC layer at the end. The dataset to fine-tune this model was sourced from [this GitHub repo from yastrebksv](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file).

The player detector module was detecting multiple people in the frame. To handle this, the court keypoints were used.
