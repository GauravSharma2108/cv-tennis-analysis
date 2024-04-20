# we will run the video frame by frame, detect and save it frame by frame
# the code for this is under utils/video_utils.py
from Utils import read_video, save_video, measure_distance, draw_player_stats, convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
from Trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court.mini_court import MiniCourt
from copy import deepcopy
import pandas as pd
import constants


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
    
    # tracking stats
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_second = (end_frame-start_frame)/24 # 24 fps

        # get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # speed of the ball shot on km/h    
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_second * 3.6

        # player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id], ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_player_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player], player_mini_court_detections[end_frame][opponent_player])
        distance_covered_by_opponent_player_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_player_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )
        speed_of_opponent = distance_covered_by_opponent_player_meters/ball_shot_time_in_second * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


    # draw bounding boxes
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_line_detector_obj.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color=(0, 255, 255))
    output_frames = draw_player_stats(output_frames,player_stats_data_df)

    # write frame number on top left of the video
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    save_video(output_frames, "Media/outputs/output_video.avi")

if __name__ == "__main__":
    main()