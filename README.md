# Analysis of a Tennis game using Computer Vision
 
## Detector engine
Ultralytics **YoloV8** was used to detect the players, and **YoloV5** to detect the tennis ball. However, the model was not able to track the tenning ball properly and hence it was fine tuned.<br>
For retraining, a pre-annotated dataset was used from [Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6#).Since this dataset only had annotations for tennis balls, the outputs of yolov8 and yolov5 were merged for player detection and tennis ball detection respectively.<br>

To detect the keypoints on the court, a pretrained ResNet50 model was used, which was fined-tuned by adding a FC layer at the end. The dataset to fine-tune this model was sourced from [this GitHub repo from yastrebksv](https://github.com/yastrebksv/TennisCourtDetector?tab=readme-ov-file).

The player detector module was detecting multiple people in the frame. To handle this, the court keypoints were used, with the logic that the detections which are closest to any one of the keypoints is a player.

## Mini court
To draw the mini-court on the video, the reference measurements of court in meters, player heights, etc. were used to convert meters to pixel distances, so that the actual court and the mini-court that was drawn for tracking were in the same ratio and in sync. Also, to track the player's and ball's position on the mini court relative to the actual frames required reference measurement calculations.

## Ball hit
Ball hits are detected based on the it's mean position along the y-axis (vertical frame direction) as a function of time. All the directional changes in the curve represents a ball hit. 

## Match statistics
The statistics are recorded and updated on ball hits, and the speeds are calculated on the basis of distance covered by objects per frame. Since the frame rate of the video is known, the distance per frame can be converted to distance per time.