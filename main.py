#------------------------------  [imports] ------------------------------------

import os
import time
import torch as T
from torch.nn import functional as F
import numpy as np
import argparse

# from pipelineB model
# from models import ResNet3D
from engagementClassifier import EngagementClassifier

# tools from pipeline A
from tools import WebcamReader, Scorer, Plotter

from finalTest import launch_finalTest

# main parameters 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Interlcutor’s attention-engagement estimation for HRI")
    
    # common parameters
    
    parser.add_argument('--useGPU',     type=bool,  default=True,   help="Usage to make compitations on tensors")
    parser.add_argument('--verbose',    type=bool,  default=True,   help="Verbose execution of the application")
    parser.add_argument('--mode',       type=str,   default="exe",  help="Choose the mode between 'exe' or 'test'")
    
    return check_arguments(parser.parse_args())

# check correctness for the parameters  
def check_arguments(args): 
    if(args.useGPU):
        try:
            assert T.cuda.is_available()
            if(args.verbose):
                print("cuda available -> " + str(T.cuda.is_available()))
                current_device = T.cuda.current_device()
                print("current device -> " + str(current_device))
                print("number of devices -> " + str(T.cuda.device_count()))
                
                try:
                    print("name device {} -> ".format(str(current_device)) + " " + str(T.cuda.get_device_name(0)))
                except Exception as e:
                    print("exception device [] -> ".format(str(current_device)) + " " + +str(e))
                
        except:
            raise NameError("Cuda is not available, please check and retry")
        
    if(args.mode):
        if not args.mode in ['exe', 'test']:
            raise ValueError("Not valid mode is selected, choose between 'exe' or 'test'")
    return args
 
 
 
# run the application
def runApplication():
    args = parse_arguments()
    
    if args is None: exit()
    
    if args.verbose:
        print("Current parameters set:")
        [print(str(k) +" -> "+ str(v)) for k,v in vars(args).items()]
    
    
    # load model trained with pipeline B
    model = EngagementClassifier(grayscale= False)
    model.loadModel(epoch = 20, path_folder="train_v2_batch4_color_depth0_epochs20_patience10_26-05-2023")
    
    if args.mode == 'exe':
        # pipeline A
        reader = WebcamReader()
        scorer = Scorer(model = model, alpha = 0.7)
        plotter = Plotter()
        reader.showAnalyzer(scorer = scorer, plotter =plotter)
    
    elif args.mode == 'test':
        launch_finalTest(model, grayscale= True, verbose= False, prog= "l")
            


if (__name__ == "__main__"):
    start_time = time.time()
    runApplication()
    time_exe = time.time() - start_time
    print(f"Execution time {time_exe}[s]")

    