# we will run the video frame by frame, detect and save it frame by frame
# the code for this is under utils/video_utils.py
from Utils import read_video, save_video
from Trackers import PlayerTracker



def main():
    
    # read video
    input_video_path = "Media/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # detect players
    player_tracker = PlayerTracker("Media/yolov8x.pt")
    player_detections = player_tracker.detect_frames(video_frames)

    # draw bounding boxes
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    save_video(video_frames, "Media/outputs/output_video.avi")

if __name__ == "__main__":
    main()