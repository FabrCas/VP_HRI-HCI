import cv2
import os
import numpy as np

class WebcamReader(object):
    def __init__(self, frame_rate = 30, resolution = 480):
        super().__init__()
        self.frame_rate = frame_rate
        self.resolution = resolution                        # higher resolution can automatically reduces the fps since usb port has a limited bandwidth
        self.path_save = "data/webcam_records"
        
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
            
            
    def open(self):
        # give name to the window
        cv2.namedWindow('Webcam')
        # Initialize the video capture object
        self.capturer = cv2.VideoCapture(0)
        self.capturer.set(cv2.CAP_PROP_FPS, self.frame_rate)
        # Set the resolution of the frames
        self.capturer.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Define the codec and create a VideoWriter object
        # self.fourcc = cv2.VideoWriter_fourcc(*'H264')                                                                                   # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.writer = cv2.VideoWriter(self.path_save + '/capture.mp4', self.fourcc, self.frame_rate, (self.width, self.height))         # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
 
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.path_save + '/capture.avi', self.fourcc, self.frame_rate, (self.width, self.height)) 
        
    
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
        
        return np.array(frames)
        
        
    def close(self):
        # Release the video capture object and close the window
        self.capturer.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
    def show(self, save= True):
        if self.capturer == None or self.writer == None:
            self.open()         
        
        # Loop through frames until the user exits
        while True:
            # Read a frame from the video capture object
            ret, frame = self.capturer.read()
            
            if not(ret):
                print("Missing frame...")
            else:
            
                # Display the frame in a window
                cv2.imshow('Webcam', frame)
                
                if save:
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
                
        self.close()
        
        
if __name__ == "__main__":
    webcam = WebcamReader(resolution= 480)
    webcam.set_frameRate(30)
    # webcam.show(save= False)
    # print("webcam fps ->", webcam.capturer.get(cv2.CAP_PROP_FPS))
    # print("webcam frame width ->", webcam.capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print("webcam frame height ->", webcam.capturer.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = webcam.readVideo("capture.avi", show=False)
    print("frames shape {}".format(frames.shape))
