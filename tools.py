import cv2
import os
import re
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pipelineA.featuresExtractors import CNN_faceExtractor, FeaturesExtractor
from pipelineA.attentionAnalyzer import AttentionAnalyzer


class WebcamReader(object):
    def __init__(self, frame_rate = 15, resolution = 480):
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
        cv2.namedWindow('Webcam')
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
                print("fps recording -> ", webcam.capturer.get(cv2.CAP_PROP_FPS))
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
        
    def showAnalyzer(self):
        self._open()
        
        analyzer = AttentionAnalyzer()
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format
            
            # flip frame
            frame = cv2.flip(frame, flipCode= 1)
            
            frame = analyzer.forward(frame, to_show=['face_box', 'yaw_face', 'debug_yaw'])
               
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
        self.plot_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.color_rf = color_rf
        self.color_values = color_values

        # max number of values on screen
        self.max_data_points = max_values
        
        # set the margin on the screen
        self.margin_y      = 50
        self.margin_x      = 75
        self.height_plot = int((self.height - (self.margin_y * 2))/3)           #height and width of just the plot, not the whole frame
        self.width_plot  = self.width - (self.margin_x * 2)
        
        # initialize the values for the plot
        self._initialize_data()
        
        # create the reference_frame
        self._build_rf()

        # start timer after creation
        self.time_start = time.time()

    
    def _build_rf(self):
        self.rf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # vertical axis
        self.rf = cv2.line(self.rf, (self.margin_x, self.height - self.margin_y), (self.margin_x, self.height - self.margin_y - self.height_plot),
                      color= self.color_rf, thickness = 5)
        
        # horizonantal axis
        self.rf = cv2.line(self.rf, (self.margin_x, self.height - self.margin_y), (self.margin_x + self.width_plot, self.height - self.margin_y),
                      color= self.color_rf, thickness = 5)
        
    
    
    def _initialize_data(self):
        
        # initialize empty list for x and y values
        self.x_data = []
        self.y_data = []
        
        # first values for x and y, and the starting one
        self.x = self.margin_x
        self.x_base = self.x
        self.y = self.plot_image.shape[0] - self.margin_y
        self.y_base = self.y
        
        # define the constant increment for the x axis
        self.x_increment =  int(self.width_plot/self.max_data_points)
        
        # define the interval
        self.x_interval = (0, self.max_data_points)
        self.y_interval = (0,1)
        
    
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
        
        
    def plot(self, frame, value, time_value = None):
        
        # clip value if necessary
        value  = self._clip(value)

        # store previous values for the plot
        y_prev = self.y
        x_prev = self.x
        
        self.x +=  self.x_increment
        self.y = self.y_base - (value * self.height_plot)
        
        # Draw the plot line on the plot image
        # update the plot
        self.plot_image = cv2.line(self.plot_image, (x_prev, y_prev), (self.x, self.y), color = self.color_values, thickness = 5)
        
        # add plot to the current frame
        # frame += self.plot_image
        
        output = self.rf + self.plot_image
        # Display the plot image with the current frame
        cv2.imshow("Real-Time Plot", output)
        
class Scorer(object):
    
    def __init__(self):
        super(Plotter).__init__()       
      
      
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
    webcam = WebcamReader(frame_rate=15, resolution= 480)
    # webcam.showFaceCNN()
    webcam.showAnalyzer()
    
    
    

        
        
    
    

    

