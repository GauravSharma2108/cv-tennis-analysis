from ultralytics import YOLO
import cv2
from typing import Dict
import pickle

class PlayerTracker:
    """
    Class to detect players in a video using YOLOv8x model
    """
    def __init__(self, model_path: str):
        """Constructor for PlayerTracker class

        Args:
            model_path (str): path to the YOLOv8x model
        """
        self.model = YOLO(model_path)

    def detect_frame(self, frame)->Dict:
        """Detect players in a single frame using YOLOv8x model, and return the bounding boxes of the players along with their track IDs

        Args:
            frame: frame of a video to detect players in

        Returns:
            dict: dictionary containing the track IDs as keys and the bounding boxes of the players as values
        """
        results = self.model.track(frame,persist=True)[0] 
        # persist=True means that the tracker will remember the object from the previous frame
        id_name_dict = results.names
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect players in multiple frames using YOLOv8x model. Calls detect_frame() for each frame.
        If read_from_stub is True, then the player detections are read from a pickle file at stub_path.
        If running for the first time, then the player detections are saved to the pickle file at stub_path if provided.

        Args:
            frames: list of frames of a video to detect players in
            read_from_stub: bool to read player detections from a pickle file
            stub_path: path to the pickle file to read player detections from or to save player detections to in the first run

        Returns:
            list: list of dictionaries containing the track IDs as keys and the bounding boxes of the players as values for each frame
        """
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections
    
    def draw_bboxes(self, frames, player_detections):
        """Draw bounding boxes around the detected players in the frames

        Args:
            frames: list of frames of a video
            player_detections: list of dictionaries containing the track IDs as keys and the bounding boxes of the players as values for each frame (output of detect_frames() method)

        Returns:
            frames: list of frames with bounding boxes drawn around the detected players
        """
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1),int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames