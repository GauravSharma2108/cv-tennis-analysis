import cv2
import constants

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 450
        self.drawing_rectangle_buffer = 50
        self.court_padding = 20
        
        self.set_canvas_bg_box_position(frame)
        self.set_mini_court_position()

    def set_canvas_bg_box_position(self, frame):
        """Set the position of the canvas background box (rectangle on which the mini court will be drawn)

        Args:
            frame: the frame on which the mini court will be drawn
        """
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.drawing_rectangle_buffer
        self.end_y = self.drawing_rectangle_height + self.drawing_rectangle_buffer
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        """
        Set the position of the mini court inside the canvas background box
        """
        self.court_start_x = self.start_x + self.court_padding
        self.court_start_y = self.start_y + self.court_padding
        self.court_end_x = self.end_x - self.court_padding
        self.court_end_y = self.end_y - self.court_padding
        self.court_drawing_width = self.court_end_x - self.court_start_x
    
    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0]*28

        # point 0
        drawing_keypoints[0], drawing_keypoints[1] = self.court_start_x, self.court_start_y
        # point 1
        drawing_keypoints[2], drawing_keypoints[3] = self.court_end_x, self.court_start_y