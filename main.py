# we will run the video frame by frame, detect and save it frame by frame
# the code for this is under utils/video_utils.py
from Utils import read_video, save_video
from Trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court.mini_court import MiniCourt


def main():
    
    # read video
    input_video_path = "Media/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # detect players and ball
    player_tracker = PlayerTracker("Models/yolov8x.pt")
    ball_tracker = BallTracker("Models/yolov5_best.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # detecting courtline keypoints
    keypoints_model_path = "Models/keypoints_model.pth"
    court_line_detector_obj = CourtLineDetector(keypoints_model_path)
    court_keypoints = court_line_detector_obj.predict(video_frames[0])

    # filter only player trackers
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Initialize MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    # draw bounding boxes
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_line_detector_obj.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color=(0, 255, 255))

    # write frame number on top left of the video
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    save_video(output_frames, "Media/outputs/output_video.avi")

if __name__ == "__main__":
    main()