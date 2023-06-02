import cv2
import os
import re
import screeninfo
import math
import time
import torch as T
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from featuresExtractors import CNN_faceExtractor, FeaturesExtractor
from attentionAnalyzer import AttentionAnalyzer

class WebcamReader(object):
    def __init__(self, frame_rate = 15, resolution = 720):
        super().__init__()
        self.frame_rate = frame_rate
        self.resolution = resolution                        # higher resolution can automatically reduces the fps since usb port has a limited bandwidth
        self.path_save = "./webcam_records"
        
        # intialize None capturer and writer 
        self.capturer = None
        self.writer = None
        self.cwd = "/".join(os.path.abspath(__file__).split("/")[:-1])
        self._create_folder()
        self.set_resolution()

        
    def set_resolution(self):
        if self.resolution == 240:
            self.width = 320
            self.height = 240
        elif self.resolution == 360:
            self.width = 640
            self.height = 360    
            
        elif self.resolution == 480:
            self.width = 640
            self.height = 480
        elif self.resolution == 720:
            self.width = 1280
            self.height = 720
        else:
            raise ValueError("Not valid resolution")

    def set_frameRate(self, frame_rate = None):
        # Set the frame rate of the video capture (not the one used from the codec)
        self.frame_rate = frame_rate
    
    def _create_folder(self):
        if not('EAI1' in os.getcwd()):
            os.chdir('Documents/EAI1')
            
        if not('data' in os.listdir()):
            os.makedirs(self.path_save)
        
        # os.chdir("data")
            
        if not(os.path.exists('data/webcam_records')):
                os.makedirs('data/webcam_records')
                
        # if not('data/webcam_records' in os.listdir()):
        #     os.mkdir("data/webcam_records")
            
    def _get_prog_captures(self):
        files = os.listdir(self.path_save)
        test_prog = -1
        for file in files:
            print(file)
            match = re.search(r"^\d*_\d*_testSet.mp4$", file)
            if match is not(None):
                print("match")
                prog = int(file.split("_")[0])
                if prog > test_prog: 
                    test_prog = prog
                    
        return test_prog+1
            
    def _open(self, candidate_id = None):
        
        # give name to the window
        # cv2.namedWindow('Webcam')

        # Initialize the video capture object
        self.capturer = cv2.VideoCapture(0)

        # Set the resolution of the frames
        self.capturer.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capturer.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        # Define the codec and create a VideoWriter object
        # self.fourcc = cv2.VideoWriter_fourcc(*'H264')                                                                                   # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.writer = cv2.VideoWriter(self.path_save + '/capture.mp4', self.fourcc, self.frame_rate, (self.width, self.height))         # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
 
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        
        if candidate_id is not(None):
            progressive = self._get_prog_captures()
            # path = self.path_save + '/' + str(progressive) + '_' + str(candidate_id) + '_' +'testSet.avi'
            path = self.path_save + '/' + str(progressive) + '_' + str(candidate_id) + '_' +'testSet.mp4'
            print(path)
            try:
                print("fps recording -> ", self.capturer.get(cv2.CAP_PROP_FPS))
                # self.writer = cv2.VideoWriter(path, self.fourcc, webcam.capturer.get(cv2.CAP_PROP_FPS), (self.width, self.height))
                self.writer = cv2.VideoWriter(path, self.fourcc, self.frame_rate, (self.width, self.height))
            except:
                print("no available path to save")

    
    def _close(self):
        
        # print recording setting
        print("webcam fps ->", self.frame_rate)
        print("webcam frame width ->", webcam.capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("webcam frame height ->", webcam.capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Release the video capture object and close the window
        self.capturer.release()
        try:
            self.writer.release()
        except:
            pass  # no instantated
        cv2.destroyAllWindows()
    
    # show methods for webcam source
    
    def show(self, save= True):
        if self.capturer == None or self.writer == None:
            if save:
                try:
                    candidate_id = int(input("Insert the candidate id\n"))
                except:
                    print("Not valid candidate id, the value must be an integer.\n The video will be not recorded")
                    candidate_id = None
            else:
                candidate_id = None
                
            self._open(candidate_id)         
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format 
            
            if frame is None: continue
            
            frame = cv2.flip(frame, flipCode= 1)
            
            if not(ret):
                print("Missing frame...")
            else:
            
                # Display the frame in a window
                cv2.imshow('Webcam', frame)
                
                if save and not(candidate_id is None):
                    self.writer.write(frame)

                # store the use input with delay
                key = cv2.waitKey(1)
                
                # Wait for the user to press 'q' to exit
                if key & 0xFF == ord('q'):
                    print('q is pressed closing all windows')
                    break
                
                # Wait for the user to press 'ESC' to exit
                if key == 27:
                    print('esc is pressed closing all windows')
                    cv2.destroyAllWindows()
                    break
                
                # Check if the user has closed the window
                if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) <1:
                    print("Closing...")
                    break
                
        self._close()
    
    def showFaceCNN(self):
        """
            stream the webcam frames detecting the most likely face on the frames using CNN model from cv2
        """
        self._open()
        
        face_extractor = CNN_faceExtractor()  
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format
            
            if frame is None: continue
            
            frame = cv2.flip(frame, flipCode= 1)
            
            frame, _ = face_extractor.drawFace(frame, display= False)
        
            if not(ret):
                print("Missing frame...")
            else:
            
                # Display the frame in a window
                cv2.imshow('Webcam', frame)
                
                # store the use input with delay
                key = cv2.waitKey(1)
                
                # Wait for the user to press 'q' to exit
                if key & 0xFF == ord('q'):
                    print('q is pressed closing all windows')
                    break
                
                # Wait for the user to press 'ESC' to exit
                if key == 27:
                    print('esc is pressed closing all windows')
                    cv2.destroyAllWindows()
                    break
                
                # Check if the user has closed the window
                if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) <1:
                    print("Closing...")
                    break
                
        self._close()
    
    def showExtractors(self):
        self._open()
        
        feature_extractor = FeaturesExtractor()
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format
            
            if frame is None: continue
            
            frame = cv2.flip(frame, flipCode= 1)
            
            frame, _, _  = feature_extractor.getLandmarks(frame, display= False, to_show=['face_box', 'eyes_lm', 'axes_eyes', 'yaw_face', 'debug_yaw'])
               
            if not(ret):
                print("Missing frame...")
            else:
            
                # Display the frame in a window
                cv2.imshow('Webcam', frame)
                
                # store the use input with delay
                key = cv2.waitKey(1)
                
                # Wait for the user to press 'q' to exit
                if key & 0xFF == ord('q'):
                    print('q is pressed closing all windows')
                    break
                
                # Wait for the user to press 'ESC' to exit
                if key == 27:
                    print('esc is pressed closing all windows')
                    cv2.destroyAllWindows()
                    break
                
                # Check if the user has closed the window
                if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) <1:
                    print("Closing...")
                    break
                
        self._close()
        
    def showAnalyzer(self, scorer, plotter, monitor_idx = 1, resolution_window = (1280,720)):
        
        self._open()
       
        analyzer = AttentionAnalyzer()
        
        screens = screeninfo.get_monitors()
        x_window = 0
        for idx, screen in enumerate(screens):
            if idx == monitor_idx:
                x_window += ((screen.width/2) - (resolution_window[0]/2))
            else:
                x_window += screen.width
                
        x_window = int(x_window)
        y_window = int((screens[monitor_idx].height/2) - (resolution_window[1]/2))
        
        cv2.namedWindow("Webcam")        # Create a named window
        cv2.moveWindow("Webcam", x_window,y_window)  # Move it to (40,30)
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format
            
            if frame is None: continue
            
            # flip frame
            frame = cv2.flip(frame, flipCode= 1)
            
            # frame, ratioX, ratioY, angleY = analyzer.forward(frame, to_show=['face_box', 'orientation_face', "gaze analytics"])
            frame, infoAnalysis  = analyzer.forward(frame, to_show=['face_box', 'orientation_face', "gaze analytics"])
            
            frame, score = scorer.forward(frame, infoAnalysis)
            
            frame = plotter.plot(frame, score)
            
            if not(ret):
                print("Missing frame...")
            else:
            
                # Display the frame in a window
                cv2.imshow('Webcam', frame)
                
                # store the use input with delay
                key = cv2.waitKey(1)
                
                # Wait for the user to press 'q' to exit
                if key & 0xFF == ord('q'):
                    print('q is pressed closing all windows')
                    break
                
                # Wait for the user to press 'ESC' to exit
                if key == 27:
                    print('esc is pressed closing all windows')
                    cv2.destroyAllWindows()
                    break
                
                # Check if the user has closed the window
                if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) <1:
                    print("Closing...")
                    break
                
        self._close()
    
    # read method for video source
    
    def readVideo(self, name_file, output_size = None, show = True):
        """
            function to read any video, choose the size and whether to show the content.
            this function returns a numpy array containing the frames
        """
        # define the empty list that will contains the frames
        frames = []
        
        if show: cv2.namedWindow('Video')

        path = self.cwd + "/" + self.path_save + "/" + name_file
        print(path)
        # Open the video file for reading
        cap = cv2.VideoCapture(path)
        
        # Define the desired output size
        if output_size == None: output_size = (self.width, self.height)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print('Error opening video file')
            exit()


        # Read and process each frame of the video
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If we have reached the end of the video, break out of the loop
            if not ret:
                print("Reached end of the video")
                break
            
            # fill the list of frames
            frames.append(frame)

            # Resize the frame to the desired output size
            frame = cv2.resize(frame, output_size)

            if show:
            # Display the resulting frame
                cv2.imshow('Video', frame)

                # Wait for a key press to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Release the video capture object and close all windows
        cap.release()
        if show: cv2.destroyAllWindows()
        
        # return the numpy array representing the video frames
        return np.array(frames)
          
class Plotter(object):
    
    def __init__(self, resolution = 720, max_values = 100, color_rf = (255,255,255), color_values = (0, 255,0)):
        super(Plotter).__init__()
        
        self.resolution = resolution
        self._set_resolution()
        self.color_rf = color_rf
        self.color_values = color_values

        # max number of values on screen
        self.max_intervals = max_values
        self.max_data_points = max_values + 1  # to have max_values intervals i neee max_values +1 data points 
        
        # set the margin on the screen
        self.margin_y      = 25
        self.margin_x      = 75
        self.height_plot = int((self.height - (self.margin_y * 2))/4)           #height and width of just the plot, not the whole frame
        self.width_plot  = self.width - (self.margin_x * 2)
        
        # initialize the values for the plot
        self._initialize_data()
    
    def _build_rf(self):
        self.rf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # vertical axis
        self.rf = cv2.line(self.rf, (self.margin_x, self.height - self.margin_y), (self.margin_x, self.height - self.margin_y - self.height_plot),
                      color= self.color_rf, thickness = 5)
        
        # horizonantal axis
        self.rf = cv2.line(self.rf, (self.margin_x, self.height - self.margin_y), (self.margin_x + self.width_plot, self.height - self.margin_y),
                      color= self.color_rf, thickness = 5)
        
        
        
    def _draw_rf(self, frame):
        # vertical axis
        self.rf = cv2.line(frame, (self.margin_x, self.height - self.margin_y), (self.margin_x, self.height - self.margin_y - self.height_plot),
                      color= self.color_rf, thickness = 5)
        
        # horizonantal axis
        self.rf = cv2.line(frame, (self.margin_x, self.height - self.margin_y), (self.margin_x + self.width_plot, self.height - self.margin_y),
                      color= self.color_rf, thickness = 5)
        
        self.rf = cv2.line(frame, (self.margin_x -10, self.height - self.margin_y - int(0.1*self.height_plot)), (self.margin_x + 10, self.height - self.margin_y - int(0.1*self.height_plot)),
                color= (0,0,255), thickness = 2)
        
        self.rf = cv2.line(frame, (self.margin_x -10, self.height - self.margin_y - int(0.3*self.height_plot)), (self.margin_x + 10, self.height - self.margin_y - int(0.3*self.height_plot)),
                color= (0,128,255), thickness = 2)
                
        self.rf = cv2.line(frame, (self.margin_x -10, self.height - self.margin_y - int(0.7*self.height_plot)), (self.margin_x + 10, self.height - self.margin_y - int(0.7*self.height_plot)),
                color= (0,255,255), thickness = 2)
                        
        self.rf = cv2.line(frame, (self.margin_x -10, self.height - self.margin_y - int(self.height_plot)), (self.margin_x + 10, self.height - self.margin_y - int(self.height_plot)),
                color= (0,255,0), thickness = 2)
            
        return frame
        
    def _initialize_data(self):
        

        
        # first values for x and y, and the starting one
        self.x = self.margin_x
        self.x_base = self.x
        self.y = self.resolution - self.margin_y
        self.y_base = self.y
        
        # initialize empty list for x and y values
        self.x_data = [self.x_base]
        self.y_data = [self.y_base]
        
        # define the constant increment for the x axis
        self.x_increment =  int(self.width_plot/self.max_intervals)
        
        # define the interval
        self.x_interval = (0, self.max_data_points)
        self.y_interval = (0,1)
        
        self.x_values = [self.x_base + i* self.x_increment for i in range(self.max_data_points)]
        
    
    def _set_resolution(self):
        if self.resolution == 240:
            self.width = 320
            self.height = 240
        elif self.resolution == 360:
            self.width = 640
            self.height = 360    
        elif self.resolution == 480:
            self.width = 640
            self.height = 480
        elif self.resolution == 720:
            self.width = 1280
            self.height = 720
        else:
            raise ValueError("Not valid resolution")
        
    def reset(self):
        self.x_data = []
        self.y_data = []
        self.time_start = time.time()
    
    def _clip(self, value):
        if value > self.y_interval[1]:   value = 1
        elif value < self.y_interval[0]: value = 0
        return value
        
    def value2color(self, value):

        if value < 0.1:
            return (0,0, 255)
        elif 0.1 <= value <= 0.3:
            return  (0,128, 255)
        elif 0.3 < value <= 0.7:
            return  (0,255, 255)
        else:
            return (0,255,0)
     
    def plot(self, frame, value, time_value = None):
        """
            value -> expected a normalized value between 0 and 1
        """
        
        if value is None:           # not enough frames to compute the value
            frame = self._draw_rf(frame)
            return frame
            
        
        # store the new data
        
        # clip value if necessary
        value_y  = self._clip(value)
        
        # compute the new y value and store
        self.y = int(self.y_base - (value_y * self.height_plot))
        self.y_data.append(self.y)
        
        # computes the next x value if necessary, if plot already filled use the defualt list of index
        
        if len(self.x_data) < self.max_data_points:
            self.x =  self.x_values[len(self.x_data)]                           # max index is 100
            self.x_data.append(self.x)
        
        else:
            while(len(self.y_data) != self.max_data_points):
                self.y_data.pop(0)                      # pop if max number of points has been reached
                self.x_data = self.x_values
            
        
           
        for i in range(len(self.x_data) -1):            
            # color = self.value2color( (-1*(self.y_data[i+1] - self.y_base))/self.height_plot )
            # frame = cv2.line(frame, (self.x_data[i],  self.y_data[i]), (self.x_data[i+1],  self.y_data[i+1]), color = color, thickness = 3)
            frame = cv2.line(frame, (self.x_data[i],  self.y_data[i]), (self.x_data[i+1],  self.y_data[i+1]), color = self.value2color(value), thickness = 3)

        frame = self._draw_rf(frame)
        
        return frame
        
class Scorer(object):
    
    def __init__(self, model = None, alpha = 0.5):
        super(Plotter).__init__()
        # self.frames = np.empty((60,3,150,120))
        self.frames = []
        # model from pipeline B
        self.model = model
        self.scores = []
        self.dims_face = (120, 150)   #(w,h)
        
        if alpha > 1 or alpha <0:
            raise ValueError("invalid value for alpha, should be between 0 and 1")
        self.alpha = alpha
        
    def normalize(self, value, min_value= 0, max_value=0.5):
        """
            functions that normalize to have a value between 0 and 1
        """
        # min_value = 0
        # max_value = 0.5
        normalized_value = (value - min_value) / (max_value - min_value)
        
        return normalized_value
    
    def label2score(self, label):
        if label == 0:
            return 0.1
        elif label == 1:
            return 0.3
        elif label == 2:
            return 0.7
        elif label == 3:
            return 1.0
        
    def score2label(self, score):
        dist = lambda x: abs(x - score)
        
        # the scores associated to the labels
        scores = [0.1, 0.3, 0.7, 1.0]
        dists = [dist(x) for x in scores]
        label = np.argmin(dists)
        return label
    
    def _centerBox(self, box):
        return (int((box[0][0] + box[1][0])/2 ), int((box[0][1] + box[1][1])/2))
    
    def _centerFrame(self, frame):
        return (int(frame.shape[1]/2), int(frame.shape[0]/2))
    
    def _clip(self, value):
        if value<=0:
            return 0
        elif value>=1:
            return 1
        else:
            return value
        
    def forward(self, frame, infoAnalyzer, to_show = ['gaze analytics'], grayscale = False):
        """
            infoAnalyzer a dictionary with the following keys:'ratioX', 'ratioY', 'angleYaw', 'limits', 'face_box'
            all scalars except for limits, a dictionary with keys:'up', 'left', 'down', 'right'
            and face_box a tuple with 2 points (tuple too) of left-top corner and bottom-right corner of the face box
        """
        # 1) if no face has been detected, returns 0
        if infoAnalyzer is None:
            return frame, 0
        
        
        #                                                   [score A]
        face_box = infoAnalyzer['face_box']
        
        # 2) store the faces in frames
        face_frame = frame[face_box[0][1]: face_box[1][1], face_box[0][0]: face_box[1][0]]
        # resize face
        face_frame = cv2.resize(face_frame, (self.dims_face[0], self.dims_face[1]), interpolation = cv2.INTER_LINEAR)
        # grayscale if requested
        if grayscale: face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        
        # plt.imshow(face_frame, cmap='gray')
        # plt.show()
        # print(face_frame.shape)
        
        self.frames.append(face_frame)
        
        if len(self.frames) < 60:
            return frame, None
        
        # if more, remove the first element
        elif len(self.frames) > 60:
            self.frames.pop(0)
            
        # print(len(self.frames))
        
        # 3) update dynamic limits
        limit_up = infoAnalyzer['limits']['up']
        limit_down = infoAnalyzer['limits']['down']
        limit_left = infoAnalyzer['limits']['left']
        limit_right = infoAnalyzer['limits']['right']
        
        
        # change the limit according to the position of the head
        
        center_face = self._centerBox(face_box)
        center_frame = self._centerFrame(frame)
        cv2.circle(frame, center_face,  radius =3, color = (255,0,0))
        cv2.circle(frame, center_frame, radius =3, color = (0,255,0))
        
        # compute the horizontal offset for the score

        max_offset_h = int(frame.shape[1]/2)
        offset_h = center_frame[0] - center_face[0]

        if not(abs(offset_h) < int(frame.shape[1]/6)):  # not centred respect the camera
            if offset_h < 0:   # face on right respect center
                # print("to the right")
                offset_h = self.normalize(abs(offset_h), 0, max_offset_h) * 3/5    # positive value between 0 and 0.5
                limit_right -= offset_h
                limit_left  -= offset_h
                # print(offset_h)
                
            else:              # face on the left respect center
                # print("to the left")
                offset_h = self.normalize(abs(offset_h), 0, max_offset_h)/2 *3/5   # positive value between 0 and 0.5
                limit_right += offset_h
                limit_left  += offset_h
                
        else: # you are central use the yaw angle
            if infoAnalyzer['angleYaw'] > 20:       # increment right bound
                # print("turned left")
                limit_right += math.sin(math.radians(infoAnalyzer['angleYaw'] -10)) # sum positive quantity
                limit_left  += math.sin(math.radians(infoAnalyzer['angleYaw'] - 10))
                
            elif infoAnalyzer['angleYaw'] < -20:    # increment left bound        
                # print("turned right")
                limit_left += math.sin(math.radians(infoAnalyzer['angleYaw'] + 10))   # sum negative quantity
                limit_right += math.sin(math.radians(infoAnalyzer['angleYaw'] + 10))
                
                
        # 4) show analytics
        if "gaze analytics" in to_show:
            gazeX_direction = infoAnalyzer['gazeX']
            gazeY_direction = infoAnalyzer['gazeY']
            cv2.putText(frame, "gaze X:" + str(gazeX_direction), (1050,20), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            cv2.putText(frame, "gaze Y:" + str(gazeY_direction), (1050,80), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            
            if not (gazeX_direction == -1):
                if gazeX_direction <=  limit_left:
                    cv2.putText(frame, "x: Left", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                elif limit_left < gazeX_direction < limit_right:
                    cv2.putText(frame, "x: Center", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                else:
                    cv2.putText(frame, "x: Right", (1050,50), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)

            if not (gazeY_direction == -1):
                if gazeY_direction <=  limit_up:
                    cv2.putText(frame, "y: Up", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                elif limit_up < gazeY_direction < limit_down:
                    cv2.putText(frame, "y: Center", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
                else:
                    cv2.putText(frame, "y: Down", (1050,110), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            
            cv2.putText(frame, "limit L: " + str(limit_left), (1050,150), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            cv2.putText(frame, "limit R: " + str(limit_right), (1050, 170), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            cv2.putText(frame, "limit U: " + str(limit_up), (1050, 190), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
            cv2.putText(frame, "limit D: " + str(limit_down), (1050, 210), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (255,255,255), thickness= 1)
        
        
        # 5) compute the error horizontal
        
        # start with the maximum score
        score_A = 1
        h_error = 0
        v_error = 0

        
        if  limit_left>= infoAnalyzer['gazeX']:
            h_error = limit_left - infoAnalyzer['gazeX']              # from 0 to 0.5     
            h_error = self.normalize(h_error, 0, 1)*2                 # normalize the error
        
        elif limit_right <= infoAnalyzer['gazeX']:
            h_error = infoAnalyzer['gazeX']  - limit_right
            # h_error = self.normalize(h_error, 0, 1)/2
            h_error = self.normalize(h_error, 0, 0.7)*2                             



        # compute the vertical error
        if  limit_up>= infoAnalyzer['gazeY']:
            v_error = limit_up - infoAnalyzer['gazeY']              # from 0 to 0.5     
            v_error = self.normalize(v_error, 0, 1)*2               # normalize the error
        
        elif limit_down <= infoAnalyzer['gazeY']:
            v_error = infoAnalyzer['gazeY']  - limit_down
            v_error = self.normalize(v_error, 0, 1)

        
        
        cv2.putText(frame, "h_error: " + str(h_error), (1050, 250), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (0,0,0), thickness= 1)
        cv2.putText(frame, "v_error: " + str(v_error), (1050, 280), fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.8, color= (0,0,0), thickness= 1)
        
        # 6) compute the error vertical
        
        # 7) penalize the score with the errors
        if h_error + v_error > 0.9:
            error = 0.9
        else:
            error = h_error + v_error
            
        # error = self.normalize(h_error + v_error, 0, 1)
        score_A -= error
        score_A = self._clip(score_A)
        
        #                                                   [score B]
        
        if self.model != None:
            
            # prepare data
            frames = np.array(self.frames)
            frames = np.transpose(frames, (3, 0, 1, 2))
              
            # forward model
            y_pred = self.model.forward(frames)[0]
            
            # print(y_pred)
            
            # get score
            score_B = self.label2score(y_pred)
            
            # print(score_B)
            
            # final weighted score
            score = self.alpha * score_A + (1-self.alpha) * score_B
        else:
            score = score_A
        
        # store the score
        self.scores.append(score)
        
        return frame, score
            
      
      
# ------------------- test functions
def test_reading(webcam):
    frames = webcam.readVideo("test_capture.avi", show=True)
    print("frames shape {}".format(frames.shape))
    
def test_capture(webcam):
    webcam.show(save= True)
    
def test_plotter():
    
    # generator of numbers to dest the plotter
    def generator_number():
        i = -1
        while True:
            i += 1
            time.sleep(1)
            yield(i)
    
    plotter = Plotter(resolution = 720)
    # Create an empty image to display the plot
    frame  = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow("Real-Time Plot", frame)
    time.sleep(1)
    
    for val in generator_number():
        plotter.plot(frame, val)
        
        # Check for key press to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
        

if __name__ == "__main__":
    webcam = WebcamReader(frame_rate=15, resolution= 720)
    scorer = Scorer()
    plotter = Plotter()
    webcam.showAnalyzer(scorer, plotter)
    
    
    

        
        
    
    

    

