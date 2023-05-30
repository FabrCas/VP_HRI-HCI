
import math
import cv2 
import numpy as np
# import features detectors
try:
    from pipelineA.featuresExtractors import FeaturesExtractor
except:
    from featuresExtractors import FeaturesExtractor


class AttentionAnalyzer():
    
    def __init__(self):
        
        # define the feature_extractor
        self.extractor = FeaturesExtractor()
        self.show_analytics =  True
        
        
        
    def forward(self, frame, to_show = None):
        """
            @ to_show: vector containing the keywords to choose what to display on the frame, the values are:
            'face_box', 'eyes_lm', 'eyes_boxes','axes_eyes','yaw_face', 'debug_yaw'
        """
        
        if to_show is None:
            new_frame, out, lm = self.extractor.getLandmarks(frame, display= False)
        else:
            new_frame, out, lm = self.extractor.getLandmarks(frame, display= False, to_show= to_show)
        
        if lm is None:          # if no face detected i have not LMs
            return new_frame
        
        # call analyzer functions
        # angle, new_frame = self.getFaceAlignment(frame, lm, to_show)
        new_frame = self.getGazeDirection(new_frame, out, lm, to_show)
        
        
        
        return new_frame
        
    def getFaceAlignment(self, frame, landmarks, to_show = ['yaw_face', 'debug_yaw'], color = (0,255,0)):
        """
            compute the face alignement from landmarks
        """
        
        # if frame and landmarks
        # compute points to infer the yaw orientatation of the face
        left_eye    = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye   = (landmarks.part(45).x, landmarks.part(45).y)
        nose_tip    = (landmarks.part(30).x, landmarks.part(30).y)
        
        # Calculate the yaw angle with simple trigonoemtry, first compute vector from midpoint eyes to nose tip, then get the angle
        midpoint_x = int((left_eye[0] + right_eye[0]) / 2)
        midpoint_y = int((left_eye[1] + right_eye[1]) / 2)
        dx = nose_tip[0] - midpoint_x
        dy = nose_tip[1] - midpoint_y
        
        alpha_angle = math.degrees(math.atan2(dy, dx)) - 90
        
        # relative reference frame
        re_segment = (right_eye[0] - midpoint_x, right_eye[1] - midpoint_y)
        
        beta_angle = math.degrees(math.atan2(re_segment[1], re_segment[0])) - alpha_angle
        
        # reference line 
        cv2.line(img = frame, pt1 = (midpoint_x -200, midpoint_y), pt2 = (midpoint_x +200, midpoint_y), color = (255,255,255), thickness=1)
        
        # compute the d magnitude        
        magnitude_d = math.hypot(dx, dy)
        
        d_proj = (int(magnitude_d * math.sin(math.radians(beta_angle))),
                         int(magnitude_d * math.cos(math.radians(beta_angle))) )
        
        nose_tip_proj = (midpoint_x + d_proj[0], midpoint_y + d_proj[1])
    
        # compute both yaw angles and show the difference if debug otherwise use the projected one
        yaw_angle_proj= math.degrees(math.atan2(d_proj[1], d_proj[0]))
        yaw_angle = math.degrees(math.atan2(dy, dx))
        
        # Draw the yaw angle
        if "yaw_face" in to_show and not 'debug_yaw' in to_show:
            cv2.putText(frame, f"Yaw Angle' : {yaw_angle_proj:.2f}", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
        elif 'debug_yaw' in to_show: 
            # show text
            cv2.putText(frame, f"Alpha Angle: {alpha_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Beta Angle : {beta_angle:.2f}",(10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw Angle  : {yaw_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw Angle' : {yaw_angle_proj:.2f}", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # show the eyes axis vector
            cv2.line(img = frame, pt1 = left_eye, pt2 = right_eye, color = color, thickness=1)
            
            # show vector alignment projected and not
            cv2.line(img = frame, pt1 = (midpoint_x, midpoint_y), pt2 = (midpoint_x + dx, midpoint_y + dy), color = color, thickness=1)
            cv2.circle(frame, (nose_tip[0], nose_tip[1]), radius= 3, color = color)
            
            cv2.line(img = frame, pt1 = (midpoint_x, midpoint_y), pt2 = (midpoint_x + d_proj[0], midpoint_y + d_proj[1]), color = (255,255,255), thickness=1)
            cv2.circle(frame, (nose_tip_proj[0], nose_tip_proj[1]), radius= 3, color = (255,255,255))
            
        return yaw_angle_proj, frame


    def getWhiteRatio(self, frame, eye_lm, eye_box, name, color = (0, 255, 0), thickness = 1, debug = True):
        
        original_frame_gray = np.copy(frame)
        original_frame_gray = cv2.cvtColor(original_frame_gray, cv2.COLOR_BGR2GRAY)
        
        
        # 1) we extract the eye drawing a polygon containing iris, sclera and pupil
        eye_points = np.array(eye_lm, dtype= np.int32)

        # 2) create the mask to isolte just the eye pixels
        mask = np.zeros(frame.shape[:2], dtype= np.uint8)       # define the empty matrix representing the mask
        # draw the eye polygon
        cv2.polylines(mask, np.int32([eye_points]), True, color = 255, thickness= 2) # one channel image, so i choose the maximum white as color
        # fill the polygon area
        cv2.fillPoly(mask, [eye_points], color = 255)
        # apply the mask to the gray original frame
        masked_frame = cv2.bitwise_and(original_frame_gray, mask)
        
        
        # 3) extract the left eye patch & segment
        eye_patch_masked = masked_frame[eye_box[1][1]:eye_box[0][1], eye_box[0][0]: eye_box[1][0]:]
        # resize the patch
        eye_patch_masked = cv2.resize(eye_patch_masked, None, fx = 5, fy= 5)
        # thresholding using binary segmenting
        # _, left_eye_patch_gray_thr = cv2.threshold(left_eye_patch, 60, 255, cv2.THRESH_BINARY)  #thrshold value 70
        _, eye_patch_masked_thr = cv2.threshold(eye_patch_masked, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # define left and right threshold
        left_patch_thr    = eye_patch_masked_thr[0: eye_patch_masked_thr.shape[0], 0: int(eye_patch_masked_thr.shape[1]/2)]
        right_patch_thr   = eye_patch_masked_thr[0: eye_patch_masked_thr.shape[0], int(eye_patch_masked_thr.shape[1]/2): eye_patch_masked_thr.shape[1]]
        
        # determine the gaze direction looking at the distribution of white pixels 
        le_left_white_pixels = cv2.countNonZero(left_patch_thr)
        le_right_white_pixels = cv2.countNonZero(right_patch_thr)
        
        
        
        
        # show section
        if debug: 
            cv2.polylines(frame, ([eye_points]), True, color = (0,255,0), thickness= thickness)
            
            if name == "left eye":
                cv2.putText(frame, "left  white sx:" + str(le_left_white_pixels), (350,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "left  white rx:" + str(le_right_white_pixels), (350,40), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            elif name == "right eye":
                cv2.putText(frame, "right white sx:" + str(le_left_white_pixels), (350,60), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "right white rx:" + str(le_right_white_pixels), (350,80), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)

            # print the thresholds
            # cv2.imshow('{} threshold'.format(name), eye_patch_masked_thr)
            # cv2.imshow('{} threshold left'.format(name), left_patch_thr)
            # cv2.imshow('{} threshold right'.format(name), right_patch_thr)
            
        return le_left_white_pixels, le_right_white_pixels

    def getGazeDirection(self, frame, out, to_show, color = (0, 255, 0), thickness = 1, debug = True):
        
        # get a copy of the original frame
        original_frame = np.copy(frame)
        
        # compute the white pixel distribution ratio after applying a mask and segmenting
        le_left_white_pixels, le_right_white_pixels = self.getWhiteRatio(frame, out['left_eye_lm'],  out['left_eye_box'],  name = "left eye", color = color, thickness = thickness,   debug = True)
        re_left_white_pixels, re_right_white_pixels = self.getWhiteRatio(frame, out['right_eye_lm'], out['right_eye_box'], name = "right eye",color = color, thickness = thickness,   debug = True)
        
        
        if le_right_white_pixels == 0:
            ratio = -1
        else:
            ratio_white_pixels = le_left_white_pixels/le_right_white_pixels  # ideally, center is around 1, if <1 looking to the left and if >1 to the right
        

            
        
        
        # same for right eye
        

        
        
        
        
        
        return frame

if __name__ == "__main__":  
    analyzer = AttentionAnalyzer()
    