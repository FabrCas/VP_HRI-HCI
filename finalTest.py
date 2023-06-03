from time import time
import os
from tqdm import tqdm
import cv2
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import random
random.seed(22)
from sklearn.metrics import precision_score, recall_score, f1_score,     \
        confusion_matrix, hamming_loss, jaccard_score, accuracy_score, mean_absolute_error, mean_absolute_percentage_error
        
from engagementClassifier import EngagementClassifier
from tools import Scorer
# from featuresExtractors import FeaturesExtractor
from attentionAnalyzer import AttentionAnalyzer



class CustomDaisee(Dataset):
    def __init__(self, path_dataset: str, grayscale: bool = False, verbose : bool = False, ):
        super(CustomDaisee).__init__()
        

        self.grayscale  = grayscale
        self.verbose    = verbose
        
        # create path
        self.path_dataset           =  path_dataset
        self.path_dataset_labels    =  os.path.join(self.path_dataset, "gt")
        self.path_dataset_video     =  os.path.join(self.path_dataset, "video")
        
        # list of files
        
        get_id_labels = lambda x: int(x.split('_')[1].replace(".json", ""))
        get_id_video  = lambda x: int(x.split('_')[1].replace(".mp4", ""))
        
        self.list_gts    =  sorted(os.listdir(self.path_dataset_labels),    key = get_id_labels)
        self.list_videos =  sorted(os.listdir(self.path_dataset_video),     key = get_id_video)
        
        self.toTensor = transforms.ToTensor()
        # self.colorJitter = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.3, hue=0.1)
      

        
    def readVideo(self, path, augment_color = False, read_RGB = False):
        """
            read a video from name file using the relative path of the dataset type
            frames are in the BGR convention
        """
        
        # define the empty list that will contains the frames
        frames = []
        
        # Open the video file for reading
        capture = cv2.VideoCapture(path)
        
        # Check if the video file was opened successfully
        if not capture.isOpened():
            print('Error opening video file')
            exit()

        # Read and process each frame of the video
        while True:
            # Read a frame from the video
            ret, frame = capture.read()   # frame -> (480, 640, 3)
            
            # If we have reached the end of the video, break out of the loop
            if not ret:
                break
            
            # sometimes last frame is None
            if frame is not None:
                    # print(frame.shape)
                    # print(frame)
                
                
                # if self.grayscale and (len(frame.shape) == 2 or frame.shape[2]==1):   # look if it's needed, i.e. v4 doesn't need
                if self.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # frame = cv2.equalizeHist(frame)                     # compute histo equalization
                else:
                    # from BGR to RGB if requested
                    if read_RGB:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # random augmentation uses 8bit int representation, values from 0 to 255
                # frame = self.random_augment(T.tensor(frame))
                
                # convert image to a tensor with values between 0 and 1, and colors channel moved: from (w,h,c) -> (c,w,h)
                frame = self.toTensor(frame)    # torch.Size([3, 480, 640])
                
                # dataugmentation altering colors
                if augment_color: frame = self.colorJitter(frame)
                
                # fill the list of frames
                frames.append(frame)

        # Release the video capture object and close all windows
        capture.release()
        
        # return the Torch  tensor representing the video frames
        video = T.stack(frames)
        
        if video.shape[0] > 300:
            video = video[:300]
        
        elif video.shape[0] < 300:  # padding with a zero frames, just one is needed and will be skipped durinh the sampling in the classifier
            n_padding = 300 - video.shape[0]
            zero_vector = T.zeros(1, video.shape[1], video.shape[2], video.shape[3])
            for i in range(n_padding):
                 # Concatenate the zero vector along dimension 0 in the last position
                video = T.cat([video, zero_vector], dim=0) 
                  
        return video
     
    def __len__(self):
        return len(self.list_gts)

    def __getitem__(self, index, show = False):
        
        # load gt
        path_gt = os.path.join(self.path_dataset_labels, self.list_gts[index])
        with open(path_gt, "r") as file:
            json_data = file.read()
        data =  json.loads(json_data)
        label = data['label']
        
        # load video
        path_video = os.path.join(self.path_dataset_video, self.list_videos[index])
        frames  = self.readVideo(path= path_video, read_RGB = False)              # torch.Size([300, 3, 480, 640]) BGR frames       
        
        if show: 
            # remember matplotib expect an image with the RGB convention! otherwise the colors are altered
            try:
                plt.imshow(frames[0].permute(1,2,0).numpy())
                plt.show()
            except:
                plt.imshow(frames[0].permute(1,2,0))
                plt.show()

        # Return the video and its label as a dictionary
        return frames, label

def saveJson(path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent= 4)
        
def computeMetrics(output, targets, scores, prog: int, save_results = True):  #labels not used
    
    if save_results: 
        # create path if doesn't exist
        path_save = os.path.join("./results", "final_results")
        if not(os.path.exists(path_save)):
            os.makedirs(path_save)
            
        
    targets_score = [label2score(target) for target in targets]
    print(targets_score)

    metrics_results = {
                        "accuracy":         accuracy_score(targets, output, normalize= True),
                        
                        
                        "precision":        precision_score(targets, output, average = "macro", zero_division=1,),    \
                        "recall":           recall_score(targets, output, average = "macro", zero_division=1),         \
                        "f1-score":         f1_score(targets, output, average= "macro", zero_division=1),            \
                        "jaccard_score":    jaccard_score(targets, output, average= "macro", zero_division=1),  \
                        "hamming_loss":     hamming_loss(targets,output),
                        "MAE":              mean_absolute_error(targets_score, scores),
                        "%MSE":             mean_absolute_percentage_error(targets_score, scores),
                                        
                        # "precision":        precision_score(targets, output, average = "micro", zero_division=1,),    \
                        # "recall":           recall_score(targets, output, average = "micro", zero_division=1),         \
                        # "f1-score":         f1_score(targets, output, average= "micro", zero_division=1),            \
                        # "jaccard_score":    jaccard_score(targets, output, average= 'micro', zero_division=1),  \
                            
                        "confusion_matrix": confusion_matrix(targets,output, labels= [0,1,2,3], normalize= None)
    
        }
    
    if save_results:
        # save on json, use a copy to change numpy array with list
        metrics_results_ = metrics_results.copy()
        metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
        saveJson(os.path.join(path_save, 'testingMetrics_' + str(prog) + '.json'), metrics_results_)
        
        # np.save(os.path.join(path_save, "metrics.npy"),metrics_results)
    
    for k,v in metrics_results.items():
        if k != "confusion_matrix":
            print("\nmetric: {}, result: {}".format(k,v))
        else:
            print("Confusion matrix")
            # for kcm,vcm in v.items():
            #     print("\nconfusion matrix for: {}".format(kcm))
            print(v)
    
    plot_cm(cm = metrics_results['confusion_matrix'], path_save = path_save, prog = str(prog), duration_timer= None)
    
    return metrics_results
    
def plot_cm(cm, path_save, prog, duration_timer = 5000):
    
    def close_event():
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    if duration_timer is not None: 
        timer = fig.canvas.new_timer(interval = duration_timer) # timer object with time interval in ms
        timer.add_callback(close_event)
    
    ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x= j, y = i, s= round(cm[i, j], 3), va='center', ha='center', size='xx-large')
                
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Targets', fontsize=18)
    plt.title('Confusion Matrix 3D ResNet', fontsize=18)
    
    
    plt.savefig(os.path.join(path_save, 'testingCM_'+ prog +'.png'))
    if duration_timer is not None:
        timer.start()
        plt.show()

def sample_frames(x, n_frames: int = 60, fps_sampling:int = 15, verbose = False): 
    """
        15 fps sampler function from a 30 fps video, 
        @ param x: video vector of the format (batch, frames, channels, width , height)
        @ param n_frames: number of frames to sample 
        @ param fps_sampling: sampling frequency of the video
        @ param verbose: perform extra print
    """
    
    # 30 frames in 15 fps corresponds to 2[s] of video
    # avoid last frame, sometimes complete black frame
    # sample the frames taking a random number from 0 to ((total_frames -1) - n_frames) and include the next n_frames.
    
    # check that is a downsample (no interpolation allowed)
    assert fps_sampling <= 30
    
    scaler = int(30/fps_sampling)
    
    if scaler* n_frames >= 300:
        return x
    else:
        total_frames = (x.shape[1] -1)
        start_frame_index = random.randint(0,(total_frames - n_frames*scaler))
        end_frame_index = start_frame_index + n_frames*scaler
        if verbose: print(f"start: {start_frame_index}, end:{end_frame_index}")
        x_sampled = x[:, start_frame_index:end_frame_index : scaler]
        
        
        if verbose: print(f"sampled {round((scaler*n_frames)/30, 4)}[s] of video")
        return x_sampled


def label2score(label):
    if label == 0:
        return 0.1
    elif label == 1:
        return 0.4
    elif label == 2:
        return 0.7
    elif label == 3:
        return 1.0

def score2label(score):
    dist = lambda x: abs(x - score)
    
    # the scores associated to the labels
    scores = [0.1, 0.3, 0.7, 1.0]
    dists = [dist(x) for x in scores]
    label = np.argmin(dists)
    return label


def launch_finalTest(model, grayscale = True, verbose = False, prog = 0):
    path_test = "./data/customDAISEE_v3/test"
    custom_ds_test= CustomDaisee(path_test) 
    test_dataloader = DataLoader(custom_ds_test, batch_size= 1, num_workers= 0, shuffle= False)
    n_steps = len(test_dataloader)
    print("number of samples in the testset {}".format(n_steps))
    
    
    alphas = [float(0.1 * i) for i in range(11)]
    
    
    # definition of tools
    analyzer = AttentionAnalyzer()
    scorer = Scorer()
    # define dimension for the faces
    dims_face = (120, 150)
    
    # define the array to store the result
    predictions = []; targets = []
    all_scores_A = []; all_scores_B = []
    
    for step_index, (frames_, label) in tqdm(enumerate(test_dataloader), total=n_steps):
        
        # if step_index == 11: break
        
        # prepare the frames
        frames = sample_frames(frames_)
        del frames_; gc.collect()
        frames = frames.numpy()
        frames = np.squeeze(frames)
        frames = np.transpose(frames, (0,2,3,1))
        # convert type image and range values, from 0-1 to 0-255
        frames = (frames* 255).astype(np.uint8)
        
        # prepare the label
        label = label.numpy().astype(np.int16)[0]
        
        # print(frames.shape)
        
        faces = []
        scores_A = []
        
        # compute score A and extract faces for the model
        for frame in frames:


            
            _ , out = analyzer.forward(frame, to_show=[])
            
            if out is None: continue  # then add padding
            
            # extract the face for model B
            face = frame[out['face_box'][0][1]: out['face_box'][1][1], out['face_box'][0][0]: out['face_box'][1][0]]
            face = cv2.resize(face, (dims_face[0], dims_face[1]), interpolation = cv2.INTER_LINEAR)
            if grayscale: face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(face)       
            
            # compute the score A and store
            score_A = scorer.forward_scoreA(frame, infoAnalyzer= out)
            # print(score_A)
            scores_A.append(score_A)

            
        
        avg_scoreA = sum(scores_A)/frames.shape[0]
        
        all_scores_A.append(avg_scoreA)
        if verbose: print("Score A: {}".format(avg_scoreA))
        

        if len(faces) < frames.shape[0]:
            while len(faces) < frames.shape[0]:
                if grayscale:
                    faces.append(np.ones((150,120)))
                else:
                    faces.append(np.ones((150,120,3)))
            
        
        faces = np.array(faces)
        faces = np.expand_dims(faces, axis=-1)
        faces = np.transpose(faces, (3, 0, 1, 2))
        
        
        # forward model
        y_pred = model.forward(faces)[0]
        
            
        # get score B
        score_B = label2score(y_pred)
        if verbose: print("Score B: {}, pred: {}",format(score_B, y_pred))
        
        all_scores_B.append(score_B)
        
        # score = alpha * score_A + (1-alpha) * score_B 
        # pred_label = score2label(score)
        # if verbose: print("Total score: {}, label: {}".format(score, pred_label))

        # store the results
        # predictions.append(pred_label)
        targets.append(label)
        
    
    for alpha in alphas:
        scores = []
        predictions = []
        for i in range(len(all_scores_A)):
            score = alpha * all_scores_A[i] + (1-alpha) * all_scores_B[i]
            scores.append(score)
            pred_label = score2label(score)
            predictions.append(pred_label)
        
        print(scores)
        print(predictions)
        print(targets)
    
                
        computeMetrics(output = predictions, targets= targets, scores = scores, prog = prog + "_" + str(alpha), save_results= True) 
        
# def launch_finalTest(model, alpha = 0.5, grayscale = True, verbose = False, prog = 0):
#     path_test = "./data/customDAISEE_v3/test"
#     custom_ds_test= CustomDaisee(path_test) 
#     test_dataloader = DataLoader(custom_ds_test, batch_size= 1, num_workers= 0, shuffle= False)
#     n_steps = len(test_dataloader)
#     print("number of samples in the testset {}".format(n_steps))
    
    
#     # definition of tools
#     analyzer = AttentionAnalyzer()
#     scorer = Scorer()
#     # define dimension for the faces
#     dims_face = (120, 150)
    
#     # define the array to store the result
#     predictions = []; targets = []; scores = []
#     for step_index, (frames_, label) in tqdm(enumerate(test_dataloader), total=n_steps):
        
#         # prepare the frames
#         frames = sample_frames(frames_)
#         del frames_; gc.collect()
#         frames = frames.numpy()
#         frames = np.squeeze(frames)
#         frames = np.transpose(frames, (0,2,3,1))
#         # convert type image and range values, from 0-1 to 0-255
#         frames = (frames* 255).astype(np.uint8)
        
#         # prepare the label
#         label = label.numpy().astype(np.int16)[0]
        
#         # print(frames.shape)
        
#         faces = []
#         scores_A = []
        
#         # compute score A and extract faces for the model
#         for frame in frames:


            
#             _ , out = analyzer.forward(frame, to_show=[])
            
#             if out is None: continue  # then add padding
            
#             # extract the face for model B
#             face = frame[out['face_box'][0][1]: out['face_box'][1][1], out['face_box'][0][0]: out['face_box'][1][0]]
#             face = cv2.resize(face, (dims_face[0], dims_face[1]), interpolation = cv2.INTER_LINEAR)
#             if grayscale: face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             faces.append(face)       
            
#             # compute the score A and store
#             score_A = scorer.forward_scoreA(frame, infoAnalyzer= out)
#             # print(score_A)
#             scores_A.append(score_A)

            
        
#         avg_scoreA = sum(scores_A)/frames.shape[0]
#         if verbose: print("Score A: {}".format(avg_scoreA))
        
#         # print(len(faces))
#         # print(faces[0].shape)
#         # for face in faces:
#         #     plt.imshow(face, cmap='gray')
#         #     plt.show()
        
        
#         if len(faces) < frames.shape[0]:
#             while len(faces) < frames.shape[0]:
#                 if grayscale:
#                     faces.append(np.ones((150,120)))
#                 else:
#                     faces.append(np.ones((150,120,3)))
            
        
#         faces = np.array(faces)
#         faces = np.expand_dims(faces, axis=-1)
#         faces = np.transpose(faces, (3, 0, 1, 2))
        
        
#         # forward model
#         y_pred = model.forward(faces)[0]
        
            
#         # get score B
#         score_B = label2score(y_pred)
#         if verbose: print("Score B: {}, pred: {}",format(score_B, y_pred))
        
        
#         score = alpha * score_A + (1-alpha) * score_B 
#         pred_label = score2label(score)
#         if verbose: print("Total score: {}, label: {}".format(score, pred_label))

#         # store the results
#         predictions.append(pred_label)
#         scores.append(score)
#         targets.append(label)
        
        
#     print(scores)
#     print(predictions)
#     print(targets)
    
            
#     computeMetrics(output = predictions, targets= targets, scores = scores, prog = prog, save_results= True)         
       
# def iter_test(model, letter:str):
#     alphas = [0.5, 0,3, 0.1, 0, 0.7, 0.9, 1]
#     for i in range(7):
#         launch_finalTest(model, alpha = alphas[i], grayscale= True, verbose= False, prog= str(i +1) + letter)
         

# tests 

# a 
# model = EngagementClassifier(grayscale= True, depth_level= 1)
# model.loadModel(epoch = 50, path_folder="train_v2_batch16_gray_depth1_epochs100_01-06-2023")
# launch_finalTest(model, grayscale= True, verbose= False, prog = "a")


# b
# model = EngagementClassifier(grayscale= True, depth_level= 1)
# model.loadModel(epoch = 55, path_folder="train_v2_batch16_gray_depth1_epochs100_01-06-2023")
# launch_finalTest(model, grayscale= True, verbose= False, prog= "b")


# c
model = EngagementClassifier(grayscale= True, depth_level= 2)
model.loadModel(epoch = 55, path_folder="train_v2_batch16_gray_depth2_epochs100_patience30_01-06-2023")
launch_finalTest(model, grayscale= True, verbose= False, prog= "c")

# d 
model = EngagementClassifier(grayscale= True, depth_level= 2)
model.loadModel(epoch = 45, path_folder="train_v2_batch16_gray_depth2_epochs80_01-06-2023")
launch_finalTest(model, grayscale= True, verbose= False, prog= "d")

# e
model = EngagementClassifier(grayscale= True, depth_level= 2)
model.loadModel(epoch = 60, path_folder="train_v2_batch16_gray_depth2_epochs80_01-06-2023")
launch_finalTest(model, grayscale= True, verbose= False, prog= "e")

# f
model = EngagementClassifier(grayscale= True, depth_level= 3)
model.loadModel(epoch = 15, path_folder="train_v2_batch8_gray_depth3_epochs100_02-06-2023")
launch_finalTest(model, grayscale= True, verbose= False, prog= "f")

# g
model = EngagementClassifier(grayscale= True, depth_level= 3)
model.loadModel(epoch = 100, path_folder="train_v2_batch8_gray_depth3_epochs100_02-06-2023")
launch_finalTest(model, grayscale= True, verbose= False, prog= "g")