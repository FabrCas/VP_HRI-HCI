
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
            'face_box', 'eyes_lm', 'eyes_boxes','axes_eyes','orientation_face', 'debug_orientation'
        """
        
        if to_show is None:
            new_frame, out, lm = self.extractor.getLandmarks(frame, display= False)
        else:
            new_frame, out, lm = self.extractor.getLandmarks(frame, display= False, to_show= to_show)
        
        if lm is None:          # if no face detected i have not LMs
            return new_frame, None
        
        # call analyzer functions
        new_frame, angle_yaw = self.getFaceOrientation(frame, lm, out, to_show)
        new_frame, gazeX, gazeY, limits = self.getGazeDirection(new_frame, out,to_show = to_show)
        
        
        info_analyzer = {'gazeX': gazeX, 'gazeY': gazeY, 'angleYaw': angle_yaw, 'limits': limits, 'face_box': out['face_box']}
        
        return new_frame, info_analyzer
        
    def getFaceOrientation(self, frame, landmarks, out, to_show = ['yaw_face', 'debug_yaw'], color = (0,255,0), pitch_estimation = False):
        """
            compute the face alignement from landmarks
        """
        
        # if frame and landmarks
        # compute points to infer the yaw orientatation of the face
        left_eye    = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye   = (landmarks.part(45).x, landmarks.part(45).y)
        nose_tip    = (landmarks.part(30).x, landmarks.part(30).y)
        chin        = (landmarks.part(8).x , landmarks.part(8).y )
        
        #                           [Yaw angle using also the roll angle]
        # Calculate the yaw angle with trigonoemtry, first compute vector from midpoint eyes to nose tip, then get the angle
        midpoint_x = int((left_eye[0] + right_eye[0]) / 2)
        midpoint_y = int((left_eye[1] + right_eye[1]) / 2)
        dx = nose_tip[0] - midpoint_x
        dy = nose_tip[1] - midpoint_y
        
        alpha_angle = math.degrees(math.atan2(dy, dx)) - 90
        
        # relative reference frame
        re_segment = (right_eye[0] - midpoint_x, right_eye[1] - midpoint_y)
        
        beta_angle = math.degrees(math.atan2(re_segment[1], re_segment[0])) - alpha_angle
        
        # compute the d magnitude        
        magnitude_d = math.hypot(dx, dy)
        
        d_proj = (int(magnitude_d * math.sin(math.radians(beta_angle))),
                         int(magnitude_d * math.cos(math.radians(beta_angle))) )
        
        nose_tip_proj = (midpoint_x + d_proj[0], midpoint_y + d_proj[1])
    
        # compute both yaw angles and show the difference if debug otherwise use the projected one
        
        # the yaw angle of the eyes-midpoint and nose project from the roll angle
        yaw_angle_proj= math.degrees(math.atan2(d_proj[1], d_proj[0])) -90
        yaw_angle = math.degrees(math.atan2(dy, dx)) -90
        
        
        if pitch_estimation:
            #                                [Pitch angle]
            # get face top
            face_top = out['face_box'][0][1]
            # Calculate the pitch angle
            pitch_angle = math.degrees(math.asin((chin[1] - nose_tip[1]) / (chin[1] - face_top))) -30
        
        # Draw the yaw angle
        if "orientation_face" in to_show and not 'debug_yaw' in to_show:
            cv2.putText(frame, f"Yaw Angle' : {yaw_angle_proj:.2f}", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            if pitch_estimation: cv2.putText(frame, f"Pitch Angle' : {pitch_angle:.2f}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
        elif 'debug_orientation' in to_show: 
            # show text
            cv2.putText(frame, f"Alpha Angle: {alpha_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Beta Angle : {beta_angle:.2f}",(10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw Angle  : {yaw_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw Angle' : {yaw_angle_proj:.2f}", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # show the eyes axis vector
            cv2.line(img = frame, pt1 = left_eye, pt2 = right_eye, color = color, thickness=1)
            
            # reference line 
            cv2.line(img = frame, pt1 = (midpoint_x -200, midpoint_y), pt2 = (midpoint_x +200, midpoint_y), color = (255,255,255), thickness=1)
            
            # show vector alignment projected and not
            cv2.line(img = frame, pt1 = (midpoint_x, midpoint_y), pt2 = (midpoint_x + dx, midpoint_y + dy), color = color, thickness=1)
            cv2.circle(frame, (nose_tip[0], nose_tip[1]), radius= 3, color = color)
            
            cv2.line(img = frame, pt1 = (midpoint_x, midpoint_y), pt2 = (midpoint_x + d_proj[0], midpoint_y + d_proj[1]), color = (255,255,255), thickness=1)
            cv2.circle(frame, (nose_tip_proj[0], nose_tip_proj[1]), radius= 3, color = (255,255,255))
            
        return frame, yaw_angle_proj 

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
        up_patch_thr      = eye_patch_masked_thr[0: int(eye_patch_masked_thr.shape[0]/2), 0: eye_patch_masked_thr.shape[1]]
        down_patch_thr    = eye_patch_masked_thr[int(eye_patch_masked_thr.shape[0]/2): eye_patch_masked_thr.shape[0], 0: eye_patch_masked_thr.shape[1]]
        # determine the gaze direction looking at the distribution of white pixels 
        
        left_white_pixels = cv2.countNonZero(left_patch_thr)
        right_white_pixels = cv2.countNonZero(right_patch_thr)
        up_white_pixels = cv2.countNonZero(up_patch_thr)
        down_white_pixels = cv2.countNonZero(down_patch_thr)
        
        cv2.polylines(frame, ([eye_points]), True, color = (0,255,0), thickness= thickness)
        # show section
        if debug: 
            cv2.polylines(frame, ([eye_points]), True, color = (0,255,0), thickness= thickness)
            
            if name == "left eye":
                # cv2.putText(frame, "left  white sx:" + str(left_white_pixels), (20,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                # cv2.putText(frame, "left  white rx:" + str(right_white_pixels), (20,40), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "left  white up:" + str(up_white_pixels), (20,60), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "left  white dw:" + str(down_white_pixels), (20,80), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            elif name == "right eye":
                # cv2.putText(frame, "right white sx:" + str(left_white_pixels), (20,640), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                # cv2.putText(frame, "right white rx:" + str(right_white_pixels), (20,660), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "right white up:" + str(up_white_pixels), (20,680), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                cv2.putText(frame, "right white dw:" + str(down_white_pixels), (20,700), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            
            # print the thresholds
            # cv2.imshow('{} threshold'.format(name), eye_patch_masked_thr)
            # cv2.imshow('{} threshold left'.format(name), left_patch_thr)
            # cv2.imshow('{} threshold right'.format(name), right_patch_thr)
            cv2.imshow('{} threshold up'.format(name), up_patch_thr)
            cv2.imshow('{} threshold down'.format(name), down_patch_thr)
            
        return left_white_pixels, right_white_pixels, up_white_pixels, down_white_pixels

    def getGazeDirection(self, frame, out, to_show, mode = 'perc_avg', color = (0, 255, 0), thickness = 1, debug = True):
        
        # get a copy of the original frame
        # original_frame = np.copy(frame)
        
        # compute the white pixel distribution ratio after applying a mask and segmenting
        le_left_white_pixels, le_right_white_pixels, le_up_white_pixels, le_down_white_pixels = self.getWhiteRatio(frame, out['left_eye_lm'],  out['left_eye_box'], name = "left eye", color = color, thickness = thickness,  debug = False)
        
        re_left_white_pixels, re_right_white_pixels, re_up_white_pixels, re_down_white_pixels= self.getWhiteRatio(frame, out['right_eye_lm'], out['right_eye_box'], name = "right eye",color = color, thickness = thickness,  debug = False)
        
        if mode == 'ratio':
            # initialize the horizontal and vertical ratio with exception value 
            ratioX_white_pixels = -1
            ratioY_white_pixels = -1
        else:
            gazeX_direction = -1
            gazeY_direction = -1
          
        # compute the horizonatal  and vertical ratio
        if mode == 'ratio':
            left_white_pixels   = le_left_white_pixels  + re_left_white_pixels
            right_white_pixels  = le_right_white_pixels + re_right_white_pixels
            up_white_pixels     = le_up_white_pixels  + re_up_white_pixels
            down_white_pixels   = le_down_white_pixels + re_down_white_pixels
            
            # cv2.putText(frame, "white sx:" + str(left_white_pixels), (1050,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            # cv2.putText(frame, "white rx:" + str(right_white_pixels), (1050,40), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
        
            # x ratio
            if right_white_pixels == 0:          # when face is no detected
                ratioX_white_pixels = -1
            else:
                ratioX_white_pixels = left_white_pixels/right_white_pixels  # ideally, center is around 1, if <1 looking to the left and if >1 to the right

            # y ratio
            if down_white_pixels == 0:
                ratioY_white_pixels = -1
            else:
                ratioY_white_pixels = up_white_pixels/down_white_pixels
            
        elif mode == 'perc_avg':
            #                                   [X gaze value]
            
            full_left = le_right_white_pixels + le_left_white_pixels
            full_right = re_right_white_pixels + re_left_white_pixels
            
            if full_left == 0 or full_right == 0: 
                gazeX_direction = -1
            
            # left eye
            le_perc_left =   le_left_white_pixels/full_left
            le_perc_right =  le_right_white_pixels/full_left
            
            # right eye
            re_perc_left =   re_left_white_pixels/full_right
            re_perc_right =  re_right_white_pixels/full_right
            
            # sum perc
            perc_left_avg = (le_perc_left + re_perc_left)/2             # min: 0, max:1
            perc_right_avg = (le_perc_right + re_perc_right)/2          # min: 0, max:1

                
            gazeX_direction = perc_left_avg - perc_right_avg                #min: -1, max: 1, positive -> right, negative -> left
            
            #                                   [Y gaze value]
              
            full_left    = le_down_white_pixels + le_up_white_pixels
            full_right   = re_down_white_pixels + re_up_white_pixels
            
            if full_left == 0 or full_right == 0: 
                gazeY_direction = -1
            
            # left eye
            le_perc_up   =   le_up_white_pixels/full_left
            le_perc_down =  le_down_white_pixels/full_left
            
            # right eye
            re_perc_up   =   re_up_white_pixels/full_right
            re_perc_down =  re_down_white_pixels/full_right
            
            # sum perc
            perc_up_avg = (le_perc_up + re_perc_up)/2                # min: 0, max:1
            perc_down_avg = (le_perc_down + re_perc_down)/2          # min: 0, max:1
            
            gazeY_direction = perc_up_avg - perc_down_avg                #min: -1, max: 1, positive -> down, negative -> up

            # cv2.putText(frame, "gaze Y: " + str(gazeY_direction), (1000,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            # cv2.putText(frame, "perc_up:  " + str(perc_up_avg), (1000,40), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            # cv2.putText(frame, "perc_down: " + str(perc_down_avg), (1000,60), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)

        
        # define the limits and round 
        if mode == 'ratio':
            
            # round the values
            ratioX_white_pixels = round(ratioX_white_pixels, 4)
            ratioY_white_pixels = round(ratioY_white_pixels, 4)
            
            limit_up = 0.25
            limit_down = 1
            limit_left = 0.2
            limit_right = 10
            
        elif mode == 'perc_avg': 
            
            # round the values
            gazeX_direction = round(gazeX_direction, 4)
            gazeY_direction = round(gazeY_direction, 4)
            
            limit_up    = -0.48
            limit_down  = -0.1
            limit_left  = -0.3
            limit_right =  0.3
        
        # save limit in a dictionary:
        limits = {'up': limit_up, 'left': limit_left, 'down': limit_down, 'right': limit_right}
        
        # if "gaze analytics" in to_show:
        
        #     if mode == 'ratio':
                # cv2.putText(frame, "ratioX:" + str(ratioX_white_pixels), (1050,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                # cv2.putText(frame, "ratioy:" + str(ratioY_white_pixels), (1050,80), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                
                # # ATTENTION: if the frame is not flipped the relation should be inverted! since the distribution behaves in a reflected way
                # # compute direction between central, left and right
                
                # if not (ratioX_white_pixels == -1):
                #     if ratioX_white_pixels <=  limit_left:
                #         cv2.putText(frame, "x: Left", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                #     elif limit_left < ratioX_white_pixels < limit_right:
                #         cv2.putText(frame, "x: Center", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                #     else:
                #         cv2.putText(frame, "x: Right", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)

                # if not (ratioY_white_pixels == -1):
                #     if ratioY_white_pixels <=  limit_up:
                #         cv2.putText(frame, "y: Up", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                #     elif limit_up < ratioY_white_pixels < limit_down:
                #         cv2.putText(frame, "y: Center", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                #     else:
                #         cv2.putText(frame, "y: Down", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                        
            # elif mode == 'perc_avg':
                

        # return section
        
        if mode == 'ratio': 
            return frame, ratioX_white_pixels, ratioY_white_pixels, limits
        
        elif mode == 'perc_avg':
            return frame, gazeX_direction, gazeY_direction, limits

if __name__ == "__main__":  
    analyzer = AttentionAnalyzer()
    