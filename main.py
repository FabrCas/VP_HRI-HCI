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





def parse_arguments():
    parser = argparse.ArgumentParser(description="Interlcutor’s attention-engagement estimation for HRI")
    
    # common parameters
    
    parser.add_argument('--useGPU', type=bool, default=True, help="Usage to make compitations on tensors")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose execution of the application")
    
    return check_arguments(parser.parse_args())
    
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
    return args
 
 
# def loadModel(model, epoch, path_folder= "./models"):
#     name_file = 'resNet3D-'+ str(epoch) +'.ckpt'
#     if not('model' in path_folder):
#         path_folder= os.path.join("./models", path_folder)
#     path_save = os.path.join(path_folder, name_file)
    
    
#     print("Loading model from: ", path_save)
#     ckpt = T.load(path_save)
#     model.load_state_dict(ckpt)
#     return model

# def forwardModel(model, x, device, verbose = False):
#     """
#         model forward, it first computes the logits, then the associated proababilities and return the most likely class.
#         @ param x: input vector in batch or not, of the shape: (channels, frames, width , height)
#     """
    
#     # wrap for the batch
#     if len(x.shape) < 5:
#         x = np.expand_dims(x, 0)
#     if len(x.shape) != 5:
#         raise ValueError("Error in the input shape for the forward")

#     # tranform in tensor if not 
#     if not T.is_tensor(x):
#         x = T.tensor(x)
    
#     # correct the dtype
#     if not (T.dtype is T.float32):
#         x = x.to(T.float32)
    
#     with T.no_grad():
#         x = x.to(device)
#         if verbose: print("input shape ->", x.shape) 
        
#         # compute logits    
#         logits  = model.forward(x)
#         if verbose: print("logits shape ->",logits.shape) 
        
#         # compute probabilities
#         probs  =  F.softmax(logits, dim = -1)               # -1, so on the last axis
#         if verbose: print("probs shape ->",probs.shape)
        
#         # get the class with max probability 
#         y_pred = T.argmax(probs, dim= -1).cpu().detach().numpy().astype(int)
#         if verbose: print("y_pred shape ->",y_pred.shape)

#     return y_pred
 
 
 
 
def runApplication():
    args = parse_arguments()
    
    if args is None: exit()
    
    if args.verbose:
        print("Current parameters set:")
        [print(str(k) +" -> "+ str(v)) for k,v in vars(args).items()]
    
    # load model trained with pipeline B
    
    # if args.useGPU:
    #     device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    # else:
    #     device = "cpu"
    
    # model: ResNet3D = ResNet3D(depth_level = 0, n_channels = 3)
    # model = loadModel(model = model, epoch = 20, path_folder="train_v2_batch4_color_depth0_epochs20_patience10_26-05-2023")

    model = EngagementClassifier(grayscale= False)
    model.loadModel(epoch = 20, path_folder="train_v2_batch4_color_depth0_epochs20_patience10_26-05-2023")
    
    # pipeline A
    reader = WebcamReader()
    scorer = Scorer(model = model, alpha = 0.7)
    plotter = Plotter()
    reader.showAnalyzer(scorer = scorer, plotter =plotter)
            


if (__name__ == "__main__"):
    start_time = time.time()
    # os.chdir("Documents/EAI1")                  # change directory to the correct project location
    runApplication()
    time_exe = time.time() - start_time
    print(f"Execution time {time_exe}[s]")

    