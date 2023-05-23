import cv2
import os 
from time import time
import numpy as np


class FastFaceExtractor(object):
    """
        Face extraction using haarcascades algorithm
    """
    
    def __init__(self , verbose = False):
        super().__init__()
        self.verbose = verbose
        # load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scale_factor = 1.1
        self.minNeighbors = 4
    
    def color2gray(self, img):
        """
            convert an color image (BGR) to grayscale
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    def getFaces(self, img: np.array, equalize = False):
        """
            image -> grey scale image
        """
        # empty list of image's faces
        faces = []
        
        original_image = np.copy(img)
        
        if img.shape[2] == 3:
            img = self.color2gray(img)
            
        # equalize grayscale image
        if equalize: img = cv2.equalizeHist(img)
        
        if self.verbose: s = time()
         
        # return the x,y cordinates of top left corner and width, height of the bounding box
        faces_box = self.face_cascade.detectMultiScale(img, scaleFactor= self.scale_factor, minNeighbors= self.minNeighbors, minSize=(30, 30))
        
        if self.verbose:
            last = time() -s 
            print(f"Face extraction in {last}[s]")
        
        for (x,y,w,h) in faces_box:
            # get face image
            face = original_image[y:y+h, x:x+w]
            faces.append(face)
        
        return faces
        
    def drawFaces(self, img, color = (0, 255,0), thickness = 2, equalize = False):
        
        original_image = np.copy(img)
        
        if img.shape[2] == 3:
            img = self.color2gray(img)
        
        if equalize: img = cv2.equalizeHist(img)
    
    
        if self.verbose: s = time()
        
        # return the x,y cordinates of top left corner and width, height of the bounding box
        faces_box = self.face_cascade.detectMultiScale(img, scaleFactor= self.scale_factor, minNeighbors= self.minNeighbors, minSize=(30, 30))
        
        if self.verbose:
            last = time() -s 
            print(f"Face extraction in {round(last, 6)}[s]")
        
        for (x,y,w,h) in faces_box:
            # draw rectanlgle on the numpy array
            cv2.rectangle(original_image, (x, y), (x + w, y + h), color, thickness)
            
        # Display the image
        cv2.imshow('image', original_image)
        cv2.waitKey(0)
             
class FastEyesDectector(object):
    """
        Eyes extraction using haarcascades algorithm
    """
    def __init__(self, verbose = False):
        super().__init__()
        self.verbose = verbose
        
        # scaleFactor: It specifies how much the image size is reduced at each image scale during the multi-scale detection process.
        # The algorithm resizes the image iteratively (pyramid downscale), and the scaleFactor controls the size reduction at each step.
        # 1.3 means reducing the image size by 30% at each scale. A smaller scaleFactor value leads to more accurate detection but requires more computational resources.
        self.scale_factor = 1.2 # 1.1
         
        # minNeighbors: It specifies the minimum number of neighboring positive detections required for an object to be considered valid.
        # During the detection process, the algorithm may produce multiple overlapping bounding boxes for the same object.
        # The minNeighbors parameter helps in filtering out false positives by requiring a minimum number of overlapping detections to consider an 
        # object as valid. Increasing minNeighbors reduces false positives but may also lead to missing detections.
        self.minNeighbors = 6 # 5
        
        
        # Load the pre-trained eye cascade classifier
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def color2gray(self, img):
        """
            convert an color image (BGR) to grayscale
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    def getEyes(self, img: np.array, equalize = False):
        """
            image -> grey scale image
        """
        # empty list of image's eyes
        eyes = []
        
        original_image = np.copy(img)
        
        if img.shape[2] == 3:
            img = self.color2gray(img)
            
        # equalize grayscale image
        if equalize: img = cv2.equalizeHist(img)
        
        if self.verbose: s = time()
         
        # return the x,y cordinates of top left corner and width, height of the bounding box
        eyes_box = self.eye_cascade.detectMultiScale(img, scaleFactor= self.scale_factor, minNeighbors= self.minNeighbors)
        
        if self.verbose:
            last = time() -s 
            print(f"Eyes extraction in {last}[s]")
        
        for (x,y,w,h) in eyes_box:
            # get eye image
            eye = original_image[y:y+h, x:x+w]
            eyes.append(eye)
        
        return eyes
        
    def drawEyes(self, img, color = (0, 255,0), thickness = 2, equalize = False):
        
        original_image = np.copy(img)
        
        if img.shape[2] == 3:
            img = self.color2gray(img)
        
        if equalize: img = cv2.equalizeHist(img)
    
    
        if self.verbose: s = time()
        
        # return the x,y cordinates of top left corner and width, height of the bounding box
        eyes_box = self.eye_cascade.detectMultiScale(img, scaleFactor= self.scale_factor, minNeighbors= self.minNeighbors)
        
        if self.verbose:
            last = time() -s 
            print(f"Eyes extraction in {round(last, 6)}[s]\n")
        
        for (x,y,w,h) in eyes_box:
            # draw rectanlgle on the numpy array
            cv2.rectangle(original_image, (x, y), (x + w, y + h), color, thickness)
            
        # Display the image
        cv2.imshow('image', original_image)
        cv2.waitKey(0)
        
# ---------------------------------------------------------------------------------------- test features extractors

def test_extractors(face_extractor, eye_extractor):
    if not('EAI1' in os.getcwd()):
        os.chdir('Documents/EAI1')
    
    path2images = "./data/test_pipelineA"
    for name_file in os.listdir(path2images):
        path = os.path.join(path2images, name_file)
        img = cv2.imread(path)
        cv2.imshow('image', img)
        cv2.waitKey(0)

        face_extractor.drawFaces(img)
        eye_extractor.drawEyes(img)
        cv2.destroyAllWindows()

        


face_extractor = FastFaceExtractor(verbose = True)
eye_extractor = FastEyesDectector(verbose =  True)
test_extractors(face_extractor, eye_extractor)