#%% 
from time import time
import multiprocessing as mp
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
import deeplake
import torch as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

"""
    The DAiSEE dataset is the first multi-label video classification dataset. It is made up of 9068 video snippets
    captured from 112 users. The videos recognize the user’s state of boredom, confusion, engagement, and frustration.
    The dataset has four levels of labels (very low, low, high, and very high) for each of the affective states.
    The labels were crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists.
"""

class CustomDaisee(Dataset):
    def __init__(self, data, vector_encoding= False, z_score_norm = False, verbose = False):
        self.data = data
        self.xs = data['video']
        self.ys = data['engagement']
        self.vector_encoding = vector_encoding
        self.z_score_norm = z_score_norm
        self.verbose = verbose
        
        # size images 
        self.w = 640
        self.h = 480
        
        self.transform = transforms.Compose([
                # i can convert to grayscale here
                transforms.ToTensor(),
                transforms.Resize((self.h,self.w), interpolation= InterpolationMode.BILINEAR, antialias= True),
                # RandAugment(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # values between -1 and 1 
                transforms.Normalize((0,0,0), (1,1,1))                     # values between 0 and 1
            ])
        
    def _preprocess(self, video):
        video_frames_ppr = []
        
        for frame in video:
            
            # new_frame = self.transform(frame)     
            
            new_frame = transforms.ToTensor()(frame) # this change the frame dimension: [colours, width, height]
            
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
                
            else:
                # default normalization 
                # new_frame = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(new_frame)   # values between -1 and 1 
                new_frame = transforms.Normalize(mean= (0,0,0), std = (1,1,1))(new_frame) # values between 0 and 1
            
              
            # TODO histogram equalization
            # convert to histogram equalization
            # new_frame = new_frame.to(T.uint8)           
            # new_frame = transforms.functional.equalize(new_frame)
            
    
            new_frame = new_frame.permute(1,2,0)
            
            video_frames_ppr.append(new_frame)      # back to the dimension: [width, height, colour ]
    
        return T.stack(video_frames_ppr)

    def label2Vector(self, label):
        label_v = np.zeros((1,4), dtype= np.int16)
        label_v[:,label[0]] = 1
        print(label_v)
        return label_v


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        x_data = self.xs[index].data()    # dim: frame, width, height, colours 
        y_data = self.ys[index].data()
        
        frames = x_data['frames']
        frames = self._preprocess(frames)
        # TODO extract face
        
        timestamps = x_data['timestamps']
        
        
        # choose if use ordinal encoding or one-hot encoding
        
        if not(self.vector_encoding):
            label = y_data['value'].astype('int16')   #.numpy(dtype = np.int16)
        else:
            label = self.label2Vector(y_data['value'])
        
        label_description = y_data["text"][0]
        
        if self.verbose:
            print(frames.shape)
            print(timestamps.shape)
            print(label)
            print(label_description)
        
        # Return the video and its label as a dictionary
        sample = {'frames': frames, 'label': label, 'timestamps': timestamps, 'label_description': label_description}
        return sample


class Dataset(object):
    
    def __init__(self):
        super().__init__()
        
        self.DATASET_PATH = "/home/faber/Documents/EAI1/data"
        
        # initialize dataloaders 
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader  = None
    
        # load data 
        self.load_dataset_offline()
        self.print_loaderCustomDaisee(self.valid_dataloader)
        
    # getters dataloaders
    
    def get_trainSet(self):
        return self.train_dataloader

    def get_validationSet(self):
        return self.valid_dataloader
    
    def get_testSet(self):
        return self.test_dataloader

    # data loading and managing

    def download_dataset(self):
        """
            downlaod dataset train, validation and test set. Sava in the data folder
        """
        self.export_env_variable()
        if "data" in os.listdir("pipelineB"):
            deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "download")
            deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "download")
            deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "download")
            
    def load_dataset_offline(self, batch_size = 1, workers = 1): #workers = 4):
        """
            load the dataset (downloading first all the data) and returns the dataloaders
        """
        try:
            self.export_env_variable()
            ds_train  = deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "local")
            # custom class for daisee dataset
            custom_ds_train= CustomDaisee(ds_train)
            train_dataloader = DataLoader(custom_ds_train, batch_size= batch_size, num_workers= workers, shuffle= True)
            
            ds_valid = deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "local")
            custom_ds_valid = CustomDaisee(ds_valid)
            valid_dataloader = DataLoader(custom_ds_valid, batch_size= batch_size, num_workers= workers, shuffle= False)
            
            ds_test  = deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "local")
            custom_ds_test = CustomDaisee(ds_test)
            test_dataloader = DataLoader(custom_ds_test, batch_size= batch_size, num_workers= workers, shuffle= False)
            
        except Exception as e:
            print(e)
            print("No data found locally, starting the download...")
            self.download_dataset()
            
            return self.load_dataset_offline(batch_size, workers)
            
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader  = test_dataloader
        
        return 
    
    def load_dataset_online(self, batch_size = 1, workers = 0):
        """
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

    # utilities and print functions
    
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
        
        print(type(data_laoder))
        for idx, data in enumerate(data_laoder):
            
            # unpack data
            video = T.squeeze(data['frames'], dim= 0)  # remove batch dimension
            label = T.squeeze(data['label'], dim = 0)
            timestamps = T.squeeze(data['timestamps'], dim=0)
            label_description = data['label_description'][0]
            
            # show types
            print(type(data['frames']))
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
                
                # print(frame)
                if show_frames:   
                    plt.imshow(frame)
                    print(frame.shape)
                    plt.show()
                    break
            
            break
        
     
custom_daisee = Dataset()