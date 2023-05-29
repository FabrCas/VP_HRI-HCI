
import math
import cv2 
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
        angle = self.getFaceAlignment(frame, lm, to_show)
        
        
        
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
            
        return yaw_angle_proj

        
    
    def getGaze(self):
        pass

if __name__ == "__main__":  
    analyzer = AttentionAnalyzer()
    