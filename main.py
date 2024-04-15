# we will run the video frame by frame, detect and save it frame by frame
# the code for this is under utils/video_utils.py
from Utils import read_video, save_video
from Trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector



def main():
    
    # read video
    input_video_path = "Media/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # detect players and ball
    player_tracker = PlayerTracker("Models/yolov8x.pt")
    ball_tracker = BallTracker("Models/yolov5_best.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    # detecting courtline keypoints
    keypoints_model_path = "Models/keypoints_model.pth"
    court_line_detector_obj = CourtLineDetector(keypoints_model_path)
    court_keypoints = court_line_detector_obj.predict(video_frames[0])

    # draw bounding boxes
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_line_detector_obj.draw_keypoints_on_video(output_frames, court_keypoints)

    save_video(output_frames, "Media/outputs/output_video.avi")

if __name__ == "__main__":
    main()