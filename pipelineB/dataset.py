
from time import time
import multiprocessing as mp
import os
from tqdm import tqdm
import gc
import cv2
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
import deeplake
import random
random.seed(22)

# torch import 
import torch as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

"""
    INFO DATASET:
    The DAiSEE dataset is the first multi-label video classification dataset. It is made up of 9068 video snippets
    captured from 112 users. The videos recognize the user’s state of boredom, confusion, engagement, and frustration.
    The dataset has four levels of labels (very low, low, high, and very high) for each of the affective states.
    The labels were crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists.
"""

class CustomDaisee(Dataset):
    def __init__(self, type_ds: str, version: str, verbose = False):
        super(CustomDaisee).__init__()
        
        self.check_args(version, type_ds)
        self.type_ds = type_ds
        self.version = version
        self.verbose = verbose
        
        # create path
        self.path_dataset           =  os.path.join("./data/customDAISEE_" + version, type_ds) # rel path from EAI1
        self.path_dataset_labels    =  os.path.join(self.path_dataset, "gt")
        self.path_dataset_video     =  os.path.join(self.path_dataset, "video")
        
        # list of files
        
        get_id_labels = lambda x: int(x.split('_')[1].replace(".json", ""))
        get_id_video  = lambda x: int(x.split('_')[1].replace(".mp4", ""))
        
        self.list_gts    =  sorted(os.listdir(self.path_dataset_labels),    key = get_id_labels)
        self.list_videos =  sorted(os.listdir(self.path_dataset_video),     key = get_id_video)
        
        # self.transform = transforms.Compose([
        #         # i can convert to grayscale here
        #         transforms.ToTensor(),
        #         transforms.Resize((self.h,self.w), interpolation= InterpolationMode.BILINEAR, antialias= True),
        #         # RandAugment(),
        #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # values between -1 and 1 
        #         transforms.Normalize((0,0,0), (1,1,1))                     # values between 0 and 1
        #     ])
        
        self.random_augment = transforms.RandAugment()
        self.toTensor = transforms.ToTensor()
      
    def check_args(self, version, type_ds):
        if (type_ds not in ['train', 'test', 'validation']) or (version not in ["v1", "v2"]):
            raise ValueError("Not valid arguments for the class {}".format(self.__class__.__name__))
        
    def readVideo(self, path, read_RGB = False):
        """
            read a video from name file using the relative path of the dataset type
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
                
                # from BGR to RGB if requested
                if read_RGB:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # TODO if training add randargument
                
                # convert image to a tensor with values between 0 and 1, and colors channel moved: from (w,h,c) -> (c,w,h)
                frame = self.toTensor(frame)    # torch.Size([3, 480, 640])
                
                # fill the list of frames
                frames.append(frame)

        # Release the video capture object and close all windows
        capture.release()
        
        # return the Torch  tensor representing the video frames
        video = T.stack(frames)
        
        if video.shape[0] > 300:
            video = video[:300]
               
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
            plt.imshow(frames[0].permute(1,2,0))
            plt.show()

        # Return the video and its label as a dictionary
        # sample = {'frames': frames, 'label': label, 'timestamps': timestamps, 'label_description': label_description}
        
        return frames, label # frames, label, timestamps, label_description

class Dataset(object):
    
    def __init__(self, batch_size = 1, use_grayscale = False, verbose = False):
        super().__init__()
        
        self.DATASET_PATH = "/home/faber/Documents/EAI1/data"
        self.batch_size = batch_size
        self.verbose = verbose
        
        
        # initialize dataloaders 
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader  = None
        
        # images processing parameters
        self.grayscale = use_grayscale
        self.w = 640
        self.h = 480
        self.vector_encoding    = False
        self.z_score_norm       = False
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    
        # load data 
        self.load_dataset_offline(workers= 0)
        
        
    # ------------------------- getters dataloaders

    def get_trainSet(self):
        return self.train_dataloader

    def get_validationSet(self):
        return self.valid_dataloader

    def get_testSet(self):
        return self.test_dataloader

    # ------------------------- pre-processing functions
    
    def _histoEq(self, frame):
        # from np array to tensor with Torch axis order
        frame = T.tensor(frame)
        frame = frame.permute(2,0,1)  # to pytorch dimensions order

        # Perform histogram equalization on each color channel
        equalized_channels = [transforms.functional.equalize(channel.unsqueeze(0)) for channel in frame]

        # Combine the equalized channels back into a color image tensor
        new_frame = T.stack(equalized_channels).squeeze(1)
        
        # new_frame = T.clamp(new_frame, 0, 1).to(T.float32)
        return new_frame
    
    # First type of pre-processing enhancing the image
    def preprocessImage(self, video, use_histo = False):
        video_frames_ppr = []
        
        for frame in video:         # rgb frame
            
            if use_histo:      
                # first perform histogram equalization 
                new_frame = self._histoEq(frame)
                
                # Resize: downsampling or upsampling (bilinear interpolation)
                new_frame = transforms.Resize((self.h,self.w), interpolation = InterpolationMode.BILINEAR, antialias= True)(new_frame)
                
                # change the range from [0-255] to [0.; 1.]
                new_frame = new_frame.float() / 255.
                
            else:
                # transform to tensor and normalize the image to have mean = 0 and std = 1.
                new_frame = transforms.ToTensor()(frame)                # this change the frame dimension: [colours width, height]
                
                # Resize: downsampling or upsampling (bilinear interpolation)
                new_frame = transforms.Resize((self.h,self.w), interpolation = InterpolationMode.BILINEAR, antialias= True)(new_frame)
                
                # gamma, saturation & sharpness correction, ...
                new_frame = transforms.functional.adjust_gamma(new_frame, gamma = 0.8, gain = 1)
                new_frame = transforms.functional.adjust_saturation(new_frame ,saturation_factor=1.2)
                new_frame = transforms.functional.adjust_sharpness(new_frame,sharpness_factor=2)
                
                # z-score normalization?
                if self.z_score_norm:
                    # Compute mean and standard deviation along each channel 
                    # TODO for all the data the mean and std
                    means = new_frame.mean(dim=(1, 2))
                    stds = new_frame.std(dim=(1, 2))

                    try:
                        new_frame = transforms.Normalize(mean= means, std = stds)(new_frame)
                    except Exception as e:
                        # handle division by zero cases
                        new_frame = transforms.Normalize(mean= (0,0,0), std = (1,1,1))(new_frame)
                    
                # default normalization for values btw 0 and 1 already performed 

            # back to the dimension: [width, height, colour]
            # new_frame = new_frame.permute(1,2,0)  
            
            # includes all the frames in a list and then stack them
            video_frames_ppr.append(new_frame)    
    
        return T.stack(video_frames_ppr)
    
    def _label2Vector(self, label):
        label_v = np.zeros((1,4), dtype= np.int32)
        label_v[:,label[0]] = 1
        return label_v
    # ------------------------- data loading and managing
    
    def save_video(self, frames, file_path): # frames dimensions [frame, width, height, colors], colors rgb
        # print("frames.shape -> ",frames.shape)
        # print(file_path)
        
        # Create a VideoWriter object
        video_writer = cv2.VideoWriter(file_path, self.fourcc, 30, (self.w, self.h))

        # Write each frame to the video file
        for frame in frames:
            # Convert the frame from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Convert the frame to uint8 if necessary
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            # Write the frame to the video file
            video_writer.write(frame)
            
        # Release the VideoWriter object and close the video file
        video_writer.release()
        
    def download_dataset(self):
        """
            downlaod dataset train, validation and test set. Sava in the data folder
        """
        self.export_env_variable()
        if not('EAI1' in os.getcwd()):
            os.chdir('Documents/EAI1')
            
        if "data" in os.listdir():
            deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "download")
            deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "download")
            deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "download")
    
    def saveCustomDataset_v1(self, type_ds, name_path = "./data/customDAISEE_v1"):
        
        print(f"Saving {type_ds} set of dataset v1...")
        
        # define the paths to saving folders
        path_save = os.path.join(name_path, type_ds)
        path_save_x = os.path.join(path_save, "video")
        path_save_y = os.path.join(path_save, "gt")
        
        if not(os.path.exists(path_save)):
            os.makedirs(path_save)
            
            if not(os.path.exists(path_save_x)):
                os.makedirs(path_save_x)
                
            if not(os.path.exists(path_save_y)):   
                os.makedirs(path_save_y)

            # indent the following codeeeee
            
            # load locally the ds from deeplake
            if type_ds == 'train':
                ds  = deeplake.load("hub://activeloop/daisee-train",        verbose= False, access_method= "local")
            elif type_ds == "validation":
                ds = deeplake.load("hub://activeloop/daisee-validation",    verbose= False, access_method= "local")
            elif type_ds == "test":
                ds = deeplake.load("hub://activeloop/daisee-test",          verbose= False, access_method= "local")
            else:
                raise ValueError("Invalid type for the dataset!")
                return 
            
            xs = ds['video']
            ys = ds['engagement']
            n_samples  = len(xs)
            
            for idx in tqdm(range(n_samples)):
                # take current x and y
                x_data = xs[idx].data()
                y_data = ys[idx].data()
                
                # get frames and do image enhancing
                frames_ = x_data['frames']      # (300, 480, 640, 3)
                
                frames = self.preprocessImage(frames_, use_histo= False)    # this move the color channel, now: [frames, color, width, height]
                frames = frames.to(T.float32)                               # torch.Size([300, 3, 480, 640]), I can change to T.float16 here
                del frames_; gc.collect()
                
                # get timestamps of video's frames
                timestamps = x_data['timestamps'].astype('float')
                
                # get label; choose the format
                if not(self.vector_encoding):
                    label = y_data['value'].astype('int64')
                    label = np.squeeze(label, axis  = 0)
                else:
                    label = self.label2Vector(y_data['value'])
                    label = np.squeeze(label, axis = 0)

                # get label descrition
                label_description = y_data["text"][0]
                
                if self.verbose:
                    print(frames.shape)
                    print(timestamps.shape)
                    print(label)
                    print(label_description)
                    
                    print(type(frames))
                    print(type(timestamps))
                    print(type(label))
                    print(type(label_description))
                    
                    frame = frames[0].permute((1,2,0)).numpy()
                    plt.imshow(frame)
                    print(frame.shape)
                    plt.show()
                
                # convert everything in list, no needed for label_description(str)
                timestamps      = list(timestamps)
                label           = label.tolist()
                    
                    
                # define the sample as a dictionary
                sample_gt = {'label': label, 'label_description': label_description, 'timestamps': timestamps}
                
                # define the relative path for the sample
                path_video  = os.path.join(path_save_x, "sample_"+ str(idx)+ ".mp4")
                path_json   = os.path.join(path_save_y, "sample_"+ str(idx)+ ".json")
                
                # save x
                frames = frames.permute(0,2,3,1).numpy()    # (300, 480, 640, 3)
                self.save_video(frames, path_video)
                
                # save y
                with open(path_json, "w") as file:
                    json.dump(sample_gt, file, indent= 4)
            
        else:
            return
          
    def saveCustomDataset_v2(self, type_ds, name_path = "./data/customDAISEE_v2"):
        # TODO
        pass
    
         
    def load_dataset_offline(self, workers = 0):
        """
            load the dataset:
            1) downloading first all the data if not locally present
            2) applying pre-processing and saving the new data if it's not already present
            3) create the pytorch Dataset object upon the new data
            4) create the dataloader for training, validation and testing
        """
        
        # prepare file system pointer  
        if not('EAI1' in os.getcwd()):
            os.chdir('Documents/EAI1')
        
        path_v1 = "./data/customDAISEE_v1"
        path_v2 = "./data/customDAISEE_v2"
        
        if not(os.path.exists(path_v1)):
            os.makedirs(path_v1)
            self.export_env_variable()
            
            try:
                self.saveCustomDataset_v1("train",      name_path= path_v1)
                self.saveCustomDataset_v1("validation", name_path= path_v1)
                self.saveCustomDataset_v1("test",       name_path= path_v1)

            except Exception as e:
                print(e)
                print("fdfd")

                # if not local data from deeplake, download it     
                if not( (os.path.exists("./EAI1/data/hub_activeloop_daisee-test")) and
                        (os.path.exists("./EAI1/data/hub_activeloop_daisee-train")) and 
                        (os.path.exists("./EAI1/data/hub_activeloop_daisee-validation")) ):
                    print("No data found locally, starting the download...")
                
                    # self.download_dataset()
                    
                    # retry after the download
                    # return self.load_dataset_offline(workers)
        
        # add elif case for v2
        
        
        # create pytorch datasets and get the dataloaders
        
        custom_ds_train= CustomDaisee(type_ds="train", version="v1")
        train_dataloader = DataLoader(custom_ds_train, batch_size= self.batch_size, num_workers= workers, shuffle= True)   
        
        custom_ds_valid= CustomDaisee(type_ds="validation", version="v1")  
        valid_dataloader = DataLoader(custom_ds_valid, batch_size= self.batch_size, num_workers= workers, shuffle= False)
        
        custom_ds_test= CustomDaisee(type_ds="test", version="v1") 
        test_dataloader = DataLoader(custom_ds_test, batch_size= self.batch_size, num_workers= workers, shuffle= False)
           
        # set the dataloaders
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader  = test_dataloader
        return 
    
    def load_dataset_online(self, batch_size = 1, workers = 0):
        """
            (Not use this since it doesn't manipulate data in the shape expected (with pre-processing) of the model)
            load the dataset (straming the data from server) and returns the dataloaders
        """
        ds_train  = deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "stream")
        train_dataloader = ds_train.pytorch(batch_size= batch_size, num_workers= workers, shuffle= True)

        ds_valid = deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "stream")
        valid_dataloader = ds_valid.pytorch(batch_size= batch_size, num_workers= workers, shuffle= False)
        
        ds_test  = deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "stream")
        test_dataloader = ds_test.pytorch(batch_size= batch_size, num_workers= workers, shuffle= False)
            
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader  = test_dataloader
        return 

    # ------------------------- utilities and print functions
    
    def export_env_variable(self, force = False):
        """
            function that set the environment variable for the download of the dataset using deeplake
        """
        
        if force:
            os.environ["DEEPLAKE_DOWNLOAD_PATH"] = self.DATASET_PATH
        else: 
            if not("DEEPLAKE_DOWNLOAD_PATH" in os.environ.keys()):
                os.environ["DEEPLAKE_DOWNLOAD_PATH"] = self.DATASET_PATH
                
    def print_env_vars(self):
        for name, value in os.environ.items():
            print(f"{name:<50}: {value:<10}")
       
    def test_num_workers(self):
        """
            simple test to choose the best number of processes to use in dataloaders
        """
        self.export_env_variable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds__test  = deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "local")
            
            for num_workers in range(2, mp.cpu_count(), 2):  
                dataloader = ds__test.pytorch(num_workers = num_workers, batch_size = 32, shuffle = True)
                start = time()
                for epoch in range(1, 3):
                    for i, data in enumerate(dataloader, 0):
                        pass
                end = time()
                print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        
    def print_loaderDaisee(self, data_laoder, show_frames = True):
        print(type(data_laoder))
        for idx, data in enumerate(data_laoder):
            video = T.squeeze(data['video'], dim= 0)  # remove batch dimension
            # print(video)
            print(video.shape)
            for frame in video:
                
                if show_frames:   # print first frame 
                    plt.imshow(frame)
                    print(frame.shape)
                    plt.show()
                    break
                
            print(data['boredom'])
            print(data['engagement'])
            print(data['confusion'])
            print(data['frustration'])
            print(data['gender'])
            
            if idx  >= 5: break
            
    def print_loaderCustomDaisee(self, data_laoder, show_frames = True):
        
        #frames, label, timestamps, label_description
        
        print(type(data_laoder))
        for idx, data in enumerate(data_laoder):       
            # unpack data
            video               = T.squeeze(data[0], dim= 0)  # remove batch dimension
            label               = T.squeeze(data[1], dim = 0)
            timestamps          = T.squeeze(data[2], dim=0)
            label_description   = data[3][0]
            
            # show types
            print(type(video))
            print(type(label))
            print(type(timestamps))
            print(type(label_description))
            
            # show shapes
            print(video.shape)
            print(label.shape)
            print(timestamps.shape)
        
            # show value label
            print(label)
            print(label_description)
            
            # show first frame and its dimension 
            for frame in video:
                
                print(frame.shape)
                
                if show_frames:  
                    frame = frame.permute(1,2,0) # re-ordering dimensions to show
                    plt.imshow(frame)
                    print(frame.shape)
                    plt.show()
                    break
            
            break
        
          
# -------------------------------------------------------------- testing
def test_fetch_speed(dataloader = None):

    if dataloader is None:
        # default deeplake dataset object to pytorch dataloader
        os.environ["DEEPLAKE_DOWNLOAD_PATH"] = "/home/faber/Documents/EAI1/data"
        ds_valid = deeplake.load("hub://activeloop/daisee-validation", verbose= False, access_method= "local")
        dataloader = ds_valid.pytorch()
    start = time()
    n_sample = 50
    for i,x in tqdm(enumerate(dataloader), total= n_sample):
        if i == 50: break
        
    t_time = time() - start
    print("\nTime elapsed {}".format(t_time))



# dataset = Dataset(batch_size=2)
# dataloader = dataset.get_validationSet()
# dataset.print_loaderCustomDaisee(dataset.get_validationSet)
# test_fetch_speed(dataloader)

