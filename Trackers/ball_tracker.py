from ultralytics import YOLO
import cv2
from typing import Dict
import pickle
import pandas as pd

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
    
    def interpolate_ball_positions(self, ball_detections):
        """Interpolate the ball positions between the frames using linear interpolation and fill the missing values using backfill for edge cases

        Args:
            ball_detections: list of dictionaries containing the track IDs as keys and the bounding boxes of the ball as values for each frame (output of detect_frames() method)

        Returns:
            list: list of dictionaries containing the track IDs as keys and the interpolated bounding boxes of the ball as values for each frame
        """
        ball_positions = [x.get(1,[]) for x in ball_detections]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate(method='linear', axis=0).copy()
        df_ball_positions = df_ball_positions.bfill().copy()
        ball_positions = [{1:x} for x in df_ball_positions.values.tolist()]
        return ball_positions
    
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits