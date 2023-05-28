import cv2
import urllib
import urllib.request
import torch as T
import os 
from time import time
import numpy as np
import dlib


# ---------- Face extractors 

class Haar_faceExtractor(object):
    """
        Fast Face extractor using haarcascades algorithm
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
        
    def drawFaces(self, img, color = (0, 255,0), thickness = 2, equalize = False, display = False):
        
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
        if display:
            cv2.imshow('image', original_image)
            cv2.waitKey(0)
        
        return original_image, faces_box
 
class CNN_faceExtractor(object):
    """
        CNN model for face extraction using caffe architecture.
        This model is based on the Caffe deep learning framework and uses a lightweight architecture designed
        for fast and efficient face detection.
        
        to use with 3 channels image
    """
    
    def __init__(self, verbose = False):
        super().__init__()
        
        self.verbose = verbose
        self.model_folder = "./models/caffe"
        self.protoPath = os.path.join(self.model_folder, "deploy.prototxt")
        self.modelPath = os.path.join(self.model_folder, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        self.download_model_files()
        self.model = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)               # model trained on color images
        
        # Check if GPU is available and set the backend and target accordingly
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # else:
        #     print("Not found GPU")

    def download_model_files(self):
        # prepare file system pointer  
        if not('EAI1' in os.getcwd()):
            os.chdir('Documents/EAI1')
            
        if not(os.path.exists(self.model_folder)):
            os.makedirs(self.model_folder)
        
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            # urllib.request.urlretrieve(proto_url, "deploy.prototxt")
            urllib.request.urlretrieve(proto_url, self.protoPath)

            # Download the pre-trained weights file
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            # urllib.request.urlretrieve(model_url, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
            urllib.request.urlretrieve(model_url, self.modelPath)
            
    def color2gray(self, img):
        """
            convert an color image (BGR) to grayscale
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def getImageFromBox(self, box, image):
        (topLeft_x, topLeft_y, bottomRight_x, bottomRight_y) = tuple(box)
        return image[topLeft_y: bottomRight_y, topLeft_x: bottomRight_x, :]
    
    def getFace(self, img, return_most_confident = True, confidence_threshold = 0.3):
        """
            get face box
        """
        # empty list for faces' boxes
        if not (return_most_confident): boxes = []
        
        # save original dimension
        (h, w) = img.shape[:2]
        
        if self.verbose: s = time()
        
        # create the blob for the model
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
        self.model.setInput(blob)
        # forward
        detections = self.model.forward()
        
        if self.verbose:
            last = time() -s 
            print(f"Face extraction in {round(last, 6)}[s]\n")
        
        # filter the boxes for confidence and draw with original dimensions
        
        if not (return_most_confident):
            for i in range(0, detections.shape[2]):
                
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  
                    box = box.astype(int)  # numpy array [top-left_x, top-left_y, bottom-right_x, bottom-right_y]
                    boxes.append(box)
            return boxes   # [self.getImageFromBox(box, img) for box in boxes], 
        
        else:
            # for i in range(0, detections.shape[2]):
            #     confidence = detections[0, 0, i, 2]
                
            confidences = [[detections[0, 0, i, 2], i] for i in range(0, detections.shape[2])]
            # print(confidences)
            max_confidence = max(confidences, key = lambda x: x[:0])
            
            if self.verbose: print("max confidence -> {}".format(max_confidence)) 
            # print(max_confidence)
            if max_confidence[0] > confidence_threshold:
                box = detections[0, 0, max_confidence[1], 3:7] * np.array([w, h, w, h])   # numpy array [top-left_x, top-left_y, bottom-right_x, bottom-right_y,]
                box = box.astype(int)
                return box   # self.getImageFromBox(box, img), 
            
    def drawFace(self, img, confidence_threshold = 0.3, return_most_confident = True, color = (0, 255,0), thickness = 2, display = False):
        """
            get face box
        """
        # save original dimension
        (h, w) = img.shape[:2]
        
        if self.verbose: s = time()
        
        # create the blob for the model
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
 
        self.model.setInput(blob)
        # forward
        detections = self.model.forward()
        
        if self.verbose:
            last = time() -s 
            print(f"Face extraction in {round(last, 6)}[s]\n")
        
        output = None
        # filter the boxes for confidence and draw with original dimensions
        if not(return_most_confident):
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype(int)
                    cv2.rectangle(img, (startX, startY), (endX, endY), color, thickness)
                    output = box
        else:
            confidences = [[detections[0, 0, i, 2], i] for i in range(0, detections.shape[2])]
            max_confidence = max(confidences, key = lambda x: x[:0])
            if max_confidence[0] > confidence_threshold:
                box = detections[0, 0, max_confidence[1], 3:7] * np.array([w, h, w, h])   # numpy array [top-left_x, top-left_y, bottom-right_x, bottom-right_y,]
                box = box.astype(int)
                (startX, startY, endX, endY) = box.astype(int)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, thickness)
                output = box
                 
        if display:
            cv2.imshow("image", img)
            cv2.waitKey(0)
            
        return img, output
        
        
                
 # -------------------------------------- Other extractors         
 
class Haar_eyesDectector(object):
    """
        Fast Eyes extractor using haarcascades algorithm
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
  
         
class FeaturesExtractor(object):
    """
        features extractor using dlib
    """
    
    def __init__(self):
        super().__init__()
        print("dlib version: {}".format(dlib.__version__))
        
        # load face detector
        self.face_extractor = dlib.get_frontal_face_detector()
        
        self.checkFileSystem()

        # define the left and right landmarks indices in a tuple representing the interval, i.e left eye indices from 36 ti 41
        self.landmarks_indices = {"left":(36, 41), "right": (42,47)}  # left and right is intented the one on the image (real is flippend)
        
        # dlib predictor for 68 facial landmarks 
        self.faceLM_predictor = dlib.shape_predictor(os.path.join(self.modelpath, self.modelfile))
    
    # -------------- auxiliar functions
    
    def checkFileSystem(self):
        if not('EAI1' in os.getcwd()):
            os.chdir('Documents/EAI1')
            
        self.modelpath = "models/dlib/"
        if not(os.path.exists(self.modelpath)):
                os.makedirs(self.modelpath)
        try:       
            self.modelfile = [name for name in os.listdir(self.modelpath) if "shape_predictor_68_face_landmarks" in name][0]
        except Exception as e:
            print(e)
            print("Please Load the predictor model from dlib site: https://www.dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            exit()
        
    def color2gray(self, img):
        """
            convert an color image (BGR) to grayscale
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def compute_midpoint(self, landmarks, is_left, frame = None, thickness = 2, color = (0,255,0)):
        
        if is_left:                 # for left eye
            # compute the left point
            lp = (landmarks.part(36).x, landmarks.part(36).y)
            # compute the right point
            rp = (landmarks.part(39).x, landmarks.part(39).y)
            line = (lp, rp)
        else:                       # for right eye
            # compute the left point
            lp = (landmarks.part(42).x, landmarks.part(42).y)
            # compute the right point
            rp = (landmarks.part(45).x, landmarks.part(45).y)
            line = (lp, rp)
        
        if frame is not None:
            cv2.line(frame, lp, rp, color, thickness)
            
    def getEyes(self, landmarks, is_left, frame = None, thickness = 1, color = (0,255,0)):
        """
            from landmarks, returns and show the box for left and right eye
        """
        # get all the points for left or right eye
        if is_left:      
            points = [(landmarks.part(idx).x, landmarks.part(idx).y)for idx in range(self.landmarks_indices["left"][0], self.landmarks_indices["left"][1] +1)]
        else:
            points = [(landmarks.part(idx).x, landmarks.part(idx).y)for idx in range(self.landmarks_indices["right"][0], self.landmarks_indices["right"][1] +1)]
        
        # print(points)
        
        # compute top left point
        xtl = min(points, key= lambda x: x[0])[0]
        ytl = max(points, key= lambda x: x[1])[1]
        
        # compute bottom right point
        xbr = max(points, key= lambda x: x[0])[0]
        ybr = min(points, key= lambda x: x[1])[1]
        
        box = ((xtl, ytl),(xbr,ybr))

        if frame is not None:
            (ptl, pbr) = box    # unpack box as top-left point and bottom-right point
            
            cv2.rectangle(frame, (ptl[0], ptl[1]), (pbr[0], pbr[1]), color, thickness)
        
        return box
            
    # -------------- feature extraction functions
    
    def getFace(self, frame, display = False, color = (0,255,0), thickness = 2):
        """
            return both the face box of the face and draw the corresponding bounding box on the face
        """
        
        if frame.shape[2] != 1: # convert in grayscale
            frame_g = self.color2gray(frame)
        
        faces = self.face_extractor(frame_g)   # return an array with all the faces (box), ordered by confidence
        
        for face in faces:
            
            # get the top-left and the bottom-right points 
            top_left_point      = (face.left(), face.top())
            bottom_right_point  = (face.right(), face.bottom())
            
            cv2.rectangle(frame, (top_left_point[0], top_left_point[1]), (bottom_right_point[0], bottom_right_point[1]), color, thickness)
        
        if display:
            cv2.imshow("image", frame)
            cv2.waitKey(0)
         
        return frame
    
    def getLandmarks(self, frame, display = False, color = (0,255,0), thickness = 2, to_show = ["face_box", 'eyes_lm','eyes_boxes']):
        """
            draws the face box, needed to find eye landmarks and returns the boxes
        """
        
        # define the output dictionary:
        output = {}
        
        if frame.shape[2] != 1: # convert in grayscale
            frame_g = self.color2gray(frame)
        
        faces = self.face_extractor(frame_g)   # return an array with all the faces (box), ordered by confidence
        
        # if is not found any face, returns the original frame and None as the output
        
        if len(faces) == 0:
            return frame, None
        
        # take the most likely face
        face = faces[0]
        
        # for face in faces:
            
        # get the top-left and the bottom-right points 
        top_left_point      = (face.left(), face.top())
        bottom_right_point  = (face.right(), face.bottom())
        
        if "face_box" in to_show:
            cv2.rectangle(frame, (top_left_point[0], top_left_point[1]), (bottom_right_point[0], bottom_right_point[1]), color, thickness)

        output['face_box'] = (top_left_point, bottom_right_point)
        
        # computes the landmarks
        landmarks = self.faceLM_predictor(frame_g, face)  # returns an object detection object
        
        left_eye_lm = []
        # draw for left eye
        for idx in range(self.landmarks_indices["left"][0], self.landmarks_indices["left"][1] +1):
            # get x and y landmark
            x = landmarks.part(idx).x
            y = landmarks.part(idx).y
            # store
            left_eye_lm.append((x,y))
            # draw it
            if "eyes_lm" in to_show: cv2.circle(frame, (x,y), radius= 2, color = (0,255,0))
        
        right_eye_lm = []
        
        # draw for right eye
        for idx in range(self.landmarks_indices["right"][0], self.landmarks_indices["right"][1] +1):
            # get x and y landmark
            x = landmarks.part(idx).x
            y = landmarks.part(idx).y
            # store
            right_eye_lm.append((x,y))
            # draw it
            if "eyes_lm" in to_show: cv2.circle(frame, (x,y), radius= 2, color = (0,255,0))
        
        output['left_eye_lm']   = left_eye_lm
        output['right_eye_lm']  = right_eye_lm
        
        # self.compute_midpoint(landmarks, is_left= True, frame= frame)
        # self.compute_midpoint(landmarks, is_left= False, frame= frame)
        
        if "eyes_boxes" in to_show:
            box_left_eye  = self.getEyes(landmarks, is_left= True, frame= frame)            # to avoid the showing of the boss, not pass the frame
            box_right_eye  = self.getEyes(landmarks, is_left= False, frame= frame)
        else:
            box_left_eye  = self.getEyes(landmarks, is_left= True)            # to avoid the showing of the boss, not pass the frame
            box_right_eye  = self.getEyes(landmarks, is_left= False)
        
        
        output['left_eye_box']  = box_left_eye
        output['right_eye_box'] = box_right_eye
        
        if display:
            cv2.imshow("image", frame)
            cv2.waitKey(0)
         
        return frame, output
        
        
# ---------------------------------------------------------------------------------------- test features extractors

def test_extractors(face_extractor = None, eye_extractor = None):
    if not('EAI1' in os.getcwd()):
        os.chdir('Documents/EAI1')
    
    path2images = "./data/test_pipelineA"
    for name_file in os.listdir(path2images):
        path = os.path.join(path2images, name_file)
        img = cv2.imread(path)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        
        if (key & 0xFF == ord('q')) or (key == 27):
            print('Test ended from user input')
            break

        if face_extractor is not None: 
            face_extractor.drawFaces(img)
        if eye_extractor is not None: 
            eye_extractor.drawEyes(img)
            
        cv2.destroyAllWindows()

def test_getFaces(face_extractor):
    if not('EAI1' in os.getcwd()):  
        os.chdir('Documents/EAI1')
    
    path2images = "./data/test_pipelineA"
    for name_file in os.listdir(path2images):
        path = os.path.join(path2images, name_file)
        img = cv2.imread(path)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
    
        if (key & 0xFF == ord('q')) or (key == 27):
            print('Test ended from user input')
            break
        
        result  = face_extractor.getFaces(img, return_most_confident=True)
        if result is list:
            for img_box in result:
                cv2.imshow("image", img_box)
                cv2.waitKey(0)
        else:
            cv2.imshow("image", result)
            cv2.waitKey(0)
        
def test_LMDetection(features_extractor: FeaturesExtractor):

    if not('EAI1' in os.getcwd()):  
        os.chdir('Documents/EAI1')

    path2images = "./data/test_pipelineA"
    for name_file in os.listdir(path2images):
        path = os.path.join(path2images, name_file)
        img = cv2.imread(path)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)

        if (key & 0xFF == ord('q')) or (key == 27):
            print('Test ended from user input')
            break
        
        frame, out = features_extractor.getLandmarks(img, to_show = ["face_box",'eyes_boxes'])
        print(out)

        cv2.imshow("image", frame)
        cv2.waitKey(0)

    
    
# face_extractor = Haar_faceExtractor(verbose = True)
# eye_extractor = Haar_eyesDectector(verbose =  True)
# test_extractors(face_extractor, eye_extractor)

# face_extractor = CNN_faceExtractor(verbose = True)
# test_extractors(face_extractor)
# test_getFaces(face_extractor)


ext = FeaturesExtractor()
test_LMDetection(ext)