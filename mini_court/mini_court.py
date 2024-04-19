import cv2
import constants
from Utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
import numpy as np

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.drawing_rectangle_buffer = 50
        self.court_padding = 20
        
        self.set_canvas_bg_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

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
    
    def convert_meters_to_pixels(self, meters):
        """Converts meters to pixels by calling the convert_meters_to_pixel_distance function from Utils

        Args:
            meters: distance in meters
        """
        return convert_meters_to_pixel_distance(meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
    
    def set_court_drawing_keypoints(self):
        """
        Set the keypoints of the mini court that will be drawn on the video
        """
        drawing_keypoints = [0]*28

        # point 0 
        drawing_keypoints[0] , drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_keypoints[2] , drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = drawing_keypoints[5] 
        # #point 4
        drawing_keypoints[8] = drawing_keypoints[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1] 
        # #point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5] 
        # #point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3] 
        # #point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7] 
        # #point 8
        drawing_keypoints[16] = drawing_keypoints[8] 
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17] 
        # #point 10
        drawing_keypoints[20] = drawing_keypoints[10] 
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_keypoints[22] = drawing_keypoints[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 
        # # #point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 
        # # #point 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21] 

        self.drawing_keypoints=drawing_keypoints

    def set_court_lines(self):
        """
        Set the lines that will be drawn on the mini court
        """
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    
    def draw_background_rectangle(self,frame):
        """Draw the background rectangle on which the mini court will be drawn

        Args:
            frame (_type_): frame for refernce size of the video

        Returns:
            frame with the background rectangle drawn
        """
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
    
    def draw_court(self,frame):
        """Draw the mini court on the video frame including the lines, net and the key points

        Args:
            frame: the frame on which the mini court will be drawn

        Returns:
            frame with the mini court drawn
        """
        for i in range(0, len(self.drawing_keypoints),2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames