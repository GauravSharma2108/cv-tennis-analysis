from ultralytics import YOLO
import cv2
from typing import Dict
import pickle

class BallTracker:
    """
    Class to detect a ball in a video using YOLOv5 model
    """
    def __init__(self, model_path: str):
        """Constructor for BallTracker class

        Args:
            model_path (str): path to the YOLOv5 model
        """
        self.model = YOLO(model_path)

    def detect_frame(self, frame)->Dict:
        """Detect a ball in a single frame using YOLOv5 model, and return the bounding box of the ball

        Args:
            frame: frame of a video to detect a ball in

        Returns:
            dict: dictionary containing the bounding box of the ball
        """
        results = self.model.predict(frame,conf=0.15)[0] # we will not track since there is only one ball
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect a ball in multiple frames using YOLOv5 model. Calls detect_frame() for each frame.
        If read_from_stub is True, then the ball detections are read from a pickle file at stub_path.
        If running for the first time, then the ball detections are saved to the pickle file at stub_path if provided.

        Args:
            frames: list of frames of a video to detect a ball in
            read_from_stub: bool to read ball detections from a pickle file
            stub_path: path to the pickle file to read ball detections from or to save ball detections to in the first run

        Returns:
            list: list of dictionaries containing the track IDs as keys and the bounding boxes of the ball as values for each frame
        """
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        ball_detections = []
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    def draw_bboxes(self, frames, ball_detections):
        """Draw bounding boxes around the detected ball in the frames

        Args:
            frames: list of frames of a video
            ball_detections: list of dictionaries containing the track IDs as keys and the bounding boxes of the ball as values for each frame (output of detect_frames() method)

        Returns:
            frames: list of frames with bounding boxes drawn around the detected ball
        """
        output_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            # draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1),int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 100, 255), 2)
            output_frames.append(frame)
        return output_frames