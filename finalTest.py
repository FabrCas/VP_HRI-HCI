from time import time
import os
from tqdm import tqdm
import cv2
import json
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
        confusion_matrix, hamming_loss, jaccard_score, accuracy_score
        
from engagementClassifier import EngagementClassifier
from tools import Scorer, Plotter
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

def saveJson(self, path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent= 4)
def _computeMetrics(output, targets, name_model, epoch: str, average = "micro", save_results = True):  #labels not used
    
    if save_results: 
        # create path if doesn't exist
        path_save = os.path.join("./results", name_model)
        if not(os.path.exists(path_save)):
            os.makedirs(path_save)

    metrics_results = {
                        "accuracy":         accuracy_score(targets, output, normalize= True),
                        "precision":        precision_score(targets, output, average = average, zero_division=1,),    \
                        "recall":           recall_score(targets, output, average = average, zero_division=1),         \
                        "f1-score":         f1_score(targets, output, average= average, zero_division=1),            \
                        # "average_precision": average_precision_score(targets, output, average= average),     \
                        "hamming_loss":     hamming_loss(targets,output),                                        \
                        "jaccard_score":    jaccard_score(targets, output, average= average, zero_division=1),  \
                        "confusion_matrix": confusion_matrix(targets,output, labels= [0,1,2,3], normalize= None)
    
        }
    
    if save_results:
        # save on json, use a copy to change numpy array with list
        metrics_results_ = metrics_results.copy()
        metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
        saveJson(os.path.join(path_save, 'testingMetrics_' + epoch + '.json'), metrics_results_)
        
        # np.save(os.path.join(path_save, "metrics.npy"),metrics_results)
    
    for k,v in metrics_results.items():
        if k != "confusion_matrix":
            print("\nmetric: {}, result: {}".format(k,v))
        else:
            print("Confusion matrix")
            # for kcm,vcm in v.items():
            #     print("\nconfusion matrix for: {}".format(kcm))
            print(v)
    
    plot_cm(cm = metrics_results['confusion_matrix'], path_save = path_save, epoch = epoch, duration_timer= None)
    
    return metrics_results
    
def plot_cm(cm, path_save, epoch, duration_timer = 5000):
    
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
    plt.savefig(os.path.join(path_save, 'testingCM_'+ epoch +'.png'))
    if duration_timer is not None: timer.start()
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

# test on v1
def launch_finalTest(batch_size = 1):
    path_test = "./data/customDAISEE_v1/test"
    custom_ds_test= CustomDaisee(path_test) 
    test_dataloader = DataLoader(custom_ds_test, batch_size= 1, num_workers= 0, shuffle= False)
    n_steps = len(test_dataloader)
    print("number of samples in the testset {}".format(n_steps))
    
    # define the array to store the result
    predictions = np.empty((0), dtype= np.int16); targets = np.empty((0), dtype= np.int16)
    for step_index, (frames, label) in tqdm(enumerate(test_dataloader), total=n_steps):
        print(step_index)
        
        
    
launch_finalTest()



def test(self, epoch_model, name_model = "final_model", verbose = True,):
        
        dataloader = self.dataset.get_testSet()
        n_steps = len(dataloader)
        # name_model = folder_model.split('models/')[1]
        print("number of samples in the testset{}".format(n_steps))
    
        # load the model and set model in evaluation mode
        try:
            self.loadModel(epoch = epoch_model, path_folder = folder_model)
        except Exception as e :
            print(e)
            print(f"Not found the model to the folder: {folder_model} of the epoch: {epoch_model}")
        self.model.eval()
        
        # define the array to store the result
        predictions = np.empty((0), dtype= np.int16); targets = np.empty((0), dtype= np.int16)
        
        
        for step_index, (frames, label) in tqdm(enumerate(dataloader), total=n_steps):
            T.cuda.empty_cache()
            if verbose: print("frames from dataloader shape ->", frames.shape)
            
            # samples from the clip
            x = self.sample_frames(frames)
            del frames; gc.collect()
            # expect input for the network: (batch_size, color_size, frames_len, width, height), so swap axis 2 and axis 1
            x = x.permute(0,2,1,3,4).to(self.device)  
            
            label = label.numpy().astype(np.int16)
            
            with T.no_grad():
                with autocast():
                    y_pred = self.forward(x)  # return a numpy array
            
            if verbose:
                print("y_pred   ->", y_pred, y_pred.shape, type(y_pred))
            
            if verbose:
                print("label    ->", label, label.shape, type(label))

            predictions = np.append(predictions, y_pred.flatten(), axis  =0)
            targets = np.append(targets, label.flatten(), axis  =0)
            
        # print(predictions.shape)
        # print(predictions)
        # print(predictions.dtype)
        
        # print(targets.shape)
        # print(targets)
        # print(targets.dtype)
        
        # compute metrics
        self._computeMetrics(output = predictions, targets= targets, name_model = name_model, epoch = str(epoch_model), save_results= True) 