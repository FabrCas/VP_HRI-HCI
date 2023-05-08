#------------------------------  [imports] ------------------------------------
import os
import time
import torch as T
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Interlcutor’s attention-engagement estimation for HRI")
    
    # common parameters
    
    parser.add_argument('--useGPU', type=bool, default=True, help="Usage to make compitations on tensors")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose execution of the application")
    
    parser.add_argument('--learning', type=bool, default=False, help="learning models, then execute")
    
    parser.add_argument('--saveDir', type=str, default='models', help='Folder in which are saved models')
    parser.add_argument('--resultDir', type=str, default='results', help='Folder in which are saved results')
    parser.add_argument('--logDir', type=str, default='logs', help='Folder in which are saved logs')
    
    
    # visual attention classifier parameters
    
    # hyperparameters (to complete)
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for learning')
    parser.add_argument('--batchSize', type=int, default=256, help='Size of batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    
    
    return check_arguments(parser.parse_args())
    
def check_arguments(args):
    
    try:
        assert args.epoch >=1
    except:
        raise NameError("epochs number must be grater than one") 
        
    try: 
        assert args.batchSize >=1
    except:
        raise NameError("Batchsize must be grater than one") 
    
    try: 
        assert args.lr >= 0
    except:
        raise NameError("learning rate must be greater than zero")
        
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
    
    # create directories whether not present
    if not os.path.exists(args.saveDir):
     os.makedirs(args.saveDir)

    if not os.path.exists(args.resultDir):
        os.makedirs(args.resultDir)
   
    # if not os.path.exists(args.logDir):
    #     os.makedirs(args.logDir)
    
    return args
 
def main():
    args = parse_arguments()
    if args is None: exit()
    
    if args.verbose:
        print("Current parameters set:")
        [print(str(k) +" -> "+ str(v)) for k,v in vars(args).items()]
        

    if args.learning:
        pass
    else:
        pass
            




if (__name__ == "__main__"):
    start_time = time.time()
    main()
    time_exe = time.time() - start_time
    print(f"Execution time {time_exe}[s]")

    