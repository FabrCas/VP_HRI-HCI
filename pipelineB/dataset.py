import deeplake
from time import time
import multiprocessing as mp
import os
import warnings

DATASET_PATH = "/home/faber/Documents/EAI1/data"

def export_env_variable(force = False):
    """
        function that set the environment variable for the download of the dataset using deeplake
    """
    
    if force:
        os.environ["DEEPLAKE_DOWNLOAD_PATH"] = DATASET_PATH
    else: 
        if not("DEEPLAKE_DOWNLOAD_PATH" in os.environ.keys()):
            os.environ["DEEPLAKE_DOWNLOAD_PATH"] = DATASET_PATH
            
def print_env_vars():
    for name, value in os.environ.items():
        print(f"{name:<50}: {value:<10}")

def download_dataset():
    """
        downlaod dataset train, validation and test set. Sava in the data folder
    """
    export_env_variable()
    if "data" in os.listdir("pipelineB"):
        deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "download")
        deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "download")
        deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "download")
        
def load_dataset(batch_size = 1, shuffle = True):
    """
        load the dataset and returns the dataloaders
    """
    try:
        export_env_variable()
        ds_train  = deeplake.load("hub://activeloop/daisee-train", verbose= True, access_method= "local")
        train_dataloader = ds_train.pytorch(num_workers = 4, batch_size = batch_size, shuffle = shuffle)
        
        ds__valid = deeplake.load("hub://activeloop/daisee-validation", verbose= True, access_method= "local")
        valid_dataloader = ds__valid.pytorch(num_workers = 4, batch_size = batch_size, shuffle = shuffle)
        
        ds__test  = deeplake.load("hub://activeloop/daisee-test", verbose= True, access_method= "local")
        test_dataloader = ds__test.pytorch(num_workers = 4, batch_size = batch_size, shuffle = False)
        
    except Exception as e:
        print(e)
        print("No data found locally, starting the download...")
        download_dataset()
        
        return load_dataset(batch_size, shuffle)
        
    return train_dataloader, valid_dataloader, test_dataloader
    
def test_num_workers():
    """
        simple test to choose the best number of processes to use in dataloaders
    """
    export_env_variable()
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
        


load_dataset()