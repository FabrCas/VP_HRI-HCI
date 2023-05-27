import cv2
import os
import re
import numpy as np
from pipelineA.featuresExtractors import Haar_faceExtractor, CNN_faceExtractor


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
        
        os.chdir("data")
        if not('webcam_records' in os.listdir()):
            os.mkdir("webcam_records")
            
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
        print("webcam fps ->", webcam.capturer.get(cv2.CAP_PROP_FPS))
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
    
    def showExtractors(self):
        self._open()
        
        # face_extractor = Haar_faceExtractor()  
        face_extractor = CNN_faceExtractor()  
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()    # BGR channels format
            
            frame, _ = face_extractor.drawFaces(frame, display= False)
                        
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
        
# ------------------- test functions
def test_reading(webcam):
    frames = webcam.readVideo("test_capture.avi", show=True)
    print("frames shape {}".format(frames.shape))
    
def test_capture(webcam):
    webcam.show(save= True)

if __name__ == "__main__":
    webcam = WebcamReader(frame_rate=15, resolution= 480)
    webcam.showExtractors()

    

