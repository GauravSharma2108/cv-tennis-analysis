import cv2

def read_video(video_path):
    """Read a video and return a list of frames

    Args:
        video_path: path to the video

    Returns:
        frames: list of frames of the video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path):
    """Save a list of frames as a video

    Args:
        frames: list of frames to save as a video
        output_path: path to save the video
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()