#%% 

import deeplake
from time import time
import multiprocessing as mp
import os
import numpy as np
import warnings
import torch as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

"""
    The DAiSEE dataset is the first multi-label video classification dataset. It is made up of 9068 video snippets
    captured from 112 users. The videos recognize the user’s state of boredom, confusion, engagement, and frustration.
    The dataset has four levels of labels (very low, low, high, and very high) for each of the affective states.
    The labels were crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists.
"""

class CustomDaisee(Dataset):
    def __init__(self, data, verbose = False):
        self.data = data
        self.xs = data['video']
        self.ys = data['engagement']
        self.verbose = verbose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        x_data = self.xs[index].data()
        y_data = self.ys[index].data()
        
        frames = x_data['frames']
        # todo extract face
        timestamps = x_data['timestamps']
        
        label = y_data['value'].astype('int16')   #.numpy(dtype = np.int16)
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
            
    def load_dataset_offline(self, batch_size = 1, workers = 4):
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
                if show_frames:   
                    plt.imshow(frame)
                    print(frame.shape)
                    plt.show()
                    break
            
            break
        
     
custom_daisee = Dataset()