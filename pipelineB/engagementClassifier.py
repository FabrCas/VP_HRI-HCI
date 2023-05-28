import numpy as np
import os 
import random
import warnings
import gc
import json

from datetime       import date
from tqdm           import tqdm
random.seed(22)
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

# classification metrics 
from sklearn.metrics import precision_score, recall_score, f1_score,     \
        confusion_matrix, hamming_loss, jaccard_score, accuracy_score
    
# torch import
import torch as T
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast

# custom classes
from models import ResNet2D, ResNet3D
from dataset import Dataset

class EngagementClassifier(nn.Module):
    # class constructor
    def __init__(self, args = None, version_dataset= "v2", video_resnet = True, grayscale = False, depth_level = 0, batch_size = 1):
        """
        @param args: list of arguments from main module 
        @param depth_level: int value to choose depth of the network
        
        """
        # correct file system position
        os.chdir("/home/faber/Documents/EAI1")
        
        # superclass call and parameters setting
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        
        # set Dataset and device
        self.batch_size = batch_size
        self.version_dataset = version_dataset
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.dataset: Dataset = Dataset(batch_size = self.batch_size, version = version_dataset, grayscale= grayscale)
        
        # Boolean flags for the model
        self.video_resnet = video_resnet
        self.grayscale = grayscale
    
        # build and load the model to GPU
        self.depth_level = depth_level
        if video_resnet:
            # use model for video clips
            if self.grayscale:
                self.model: ResNet3D = ResNet3D(depth_level = depth_level, n_channels = 1)
            else:
                self.model: ResNet3D = ResNet3D(depth_level = depth_level, n_channels = 3)
        else:
            # use model for single frames
            if self.grayscale:
                self.model: ResNet2D = ResNet2D(depth_level = depth_level, n_channels =1)
            else:
                self.model: ResNet2D = ResNet2D(depth_level = depth_level, n_channels = 3)
            
        self.model.to(self.device)
        self.init_weights_kaimingNormal()
        self.class_weights = T.tensor(self.compute_class_weights()).to(self.device)
        
        # self.learning_weights = [161.235, 25.617, 2.069, 2.121]   # weights for v1
        # self.learning_weights = [22.588, 15.059, 2.246, 2.252]    # weights for v2
        
        
        # define learning functions
        """ 
         softmax produces a probability distribution over classes, while log softmax transforms
         the softmax probabilities into a logarithmic scale for numerical stability during 
        """
        self.output_af  = F.softmax             # F.log_softmax
        self.loss_f     = F.cross_entropy       # nn.CrossEntropyLoss
        
        # learning parameters
        self.lr = 1e-4
        self.n_epochs = 10
        self.weight_decay = 0.001   # L2 regularization term 
        self.patience = 5           # patience for earling stopping 
        
        #TODO grid search for the parameters
        
        
    # ---------------- [initialization functions]
    
    def init_weights_normal(self):
        print("Initializing weights using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.normal_(param, mean=0, std=0.01) 
                
    def init_weights_kaimingNormal(self):
        # Initialize the weights  using He initialization
        print("Weights initialization using kaiming Normal")
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    # ---------------- [data processing functions]
    
    def sample_frames(self, x, n_frames: int = 30, fps_sampling:int = 15, verbose = False): 
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
    
    def compute_class_weights(self, verbose = False):
        
        print("Computing class weights for the training set")
        # get train dataset using single mini-batch size
        laoder = self.dataset.get_dataloaderLabels(type_ds= "train", version= self.version_dataset)
        
        # compute occurrences of labels
        class_freq={}
        total = len(laoder)
        for y in laoder:
            l = y.item()
            if l not in class_freq.keys():
                class_freq[l] = 1
            else:
                class_freq[l] = class_freq[l]+1
        if verbose: print("class_freq -> ", class_freq)
        
        # compute the weights   
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(total/freq)

        print("Class_weights-> ", class_weights)
        return class_weights
    
    def focal_loss(self, y_pred, y_true, alpha=None, gamma=2, reduction='sum'):
        """
            focal loss implementation to handle to problem of unbalanced classes
            y_pred -> logits from the model
            y_true -> ground truth label for the sample (no one-hot encoding)
            alpha -> weights for the classes
            gamma -> parameter controls the rate at which the focal term decreases with increasing predicted probability
            reduction -> choose between sum or mean to reduce over results in a batch
        """

        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = T.exp(-ce_loss)

        if alpha is not None:
            # Apply class-specific alpha weights
            alpha = alpha.to(y_pred.device)
            focal_loss = alpha[y_true] * (1 - pt) ** gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** gamma * ce_loss
            
        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss
    
    # ---------------- [Train, Validation &  Test functions]

    def valid(self, loader, epoch):
        """
            validation that compute the accuracy of the model in the current epoch
        """
        
        print (f"Validation for the epoch: {epoch} ...")
        # define array for validation loss story and prediction counters to compute the accuracy
        # losses = []
        correct_predictions = 0
        num_predictions = 0 
        
        
        for (frames, label) in tqdm(loader):
            
            # 1) get x and y data
            x = self.sample_frames(frames, n_frames= 60); del frames; gc.collect()
            x = x.permute(0,2,1,3,4).to(self.device)  
            y = label.to(self.device)
            
            with T.no_grad():
                with autocast():
                    logits  = self.model.forward(x)
                    
                    probs   = self.output_af(logits, dim = -1) 
                    y_pred  = T.argmax(probs, dim= -1).cpu().numpy().astype(int)
                
                    # loss    = self.focal_loss(y_pred = logits, y_true = y, alpha=self.class_weights ,reduction= 'sum')
        
            y = y.cpu().numpy()  
            # losses.append(loss.item())
            
            correct_predictions += (y_pred == y).sum()
            num_predictions += y_pred.shape[0]
            
                
        # mean_loss = sum(losses)/len(losses)
        accuracy = correct_predictions / num_predictions
        
        # return mean_loss, accuracy
        return accuracy
        

    def train(self, name_model = "test", save_model = False, verbose = True, early_stopping = True):
        
        # get the train and validation dataloaders
        train_dataloader = self.dataset.get_trainSet()
        valid_dataloader = self.dataset.get_validationSet()
        
        
        # get number of steps for epoch and current date in format "DD-MM-YYYY" for the folder name
        n_steps = len(train_dataloader)
        date_ = date.today().strftime("%d-%m-%Y")
        print("number of steps per epoch: {}".format(n_steps))
        
        # set the mode
        self.model.train()
        
        # define optimizer, scheduler & scaler
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay,)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        scaler = GradScaler()
        
        # initialize list to store the losses of each epoch
        loss_epochs = []
        
        # initialize list to store the losses from the validation
        valid_history = []
        
        # intialize to zero the counter for early stopping using patience
        patience_counter = 0
        
        # frequency used to save the model every "freq_save" epochs
        freq_save = 5
        
        # frequency used to rint and show loss every "loss_steps_freq" steps
        loss_steps_freq = 50
        
        # boolean flag indicing if the corrent epoch model has been savec
        saved: bool = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            
            for epoch_index in range(self.n_epochs) :
                
                print(f"\n             [Epoch{epoch_index+1}]             \n")
                
                # cumulative loss for the current epoch
                loss_epoch = 0
                
                # cumulative loss for print, reset every "loss_steps_freq" steps
                tmp_loss = 0
                
                # list of losses, saved before resetting tmp_loss doing the average over "loss_steps_freq" steps
                loss_steps = []
                
                # boolean flag to guarantee the saving of the model at final step
                saved = False
                
                for step_index, (frames, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                    
                    # 1) get x and y data
                    x = self.sample_frames(frames, n_frames= 60)
                    del frames; gc.collect()
                    # expect input for the network: (batch_size, color_size, frames_len, width, height), so swap axis 2 and axis 1
                    x = x.permute(0,2,1,3,4).to(self.device)  
                    x.requires_grad_(True)
                    
                    y = label.to(self.device)
                    
                    if verbose: print("x -> ", x.shape, x.dtype)
                    if verbose: print("y -> ", y, y.shape, y.dtype)
                    
                    # 2) zeroing grad and cache
                    T.cuda.empty_cache()    
                    optimizer.zero_grad()
                    
                    
                    # 3) compute forward and loss
                    with autocast():
                        logits  = self.model.forward(x)
                        if verbose: print("logits ->", logits,  type(logits), logits.shape)
                        
                        # loss = self.loss_f(logits, y, reduction= 'sum', weight= self.class_weights)
                        loss = self.focal_loss(y_pred = logits, y_true = y, alpha=self.class_weights ,reduction= 'sum')
                        
                        if verbose: print("loss ->", loss, type(loss))
                        
                        # probs  = self.output_af(logits, dim = 1)    # no used here, but needed for the full forward
                        # if verbose: print("probabilities ->", probs,  type(probs), probs.shape)
                    
                    loss_epoch += loss.item()         
                    tmp_loss += loss.item()
                    # 4) update step
                    
                    # backpropagation
                    scaler.scale(loss).backward()
                    
                    # clip gradients
                    T.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # compute updates on weights
                    scaler.step(optimizer)
                    
                    # update scaler
                    scaler.update()
                    
                    # update schedulers
                    scheduler.step()
                    
                    # print info every 50 steps 
                    if ((step_index+1)%loss_steps_freq)== 0:
                        
                        print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.3f}'
                                .format(epoch_index+1, self.n_epochs, str(step_index+1).zfill(3), str(n_steps).zfill(3),
                                        scheduler.get_last_lr()[0], \
                                        tmp_loss/loss_steps_freq))
                        
                        loss_steps.append(tmp_loss/loss_steps_freq)
                        tmp_loss = 0  # resetting the periodic loss for steps
                        
                
                # print info and plot of loss for the current epoch  
                self.plot_loss(loss_array= loss_steps, epoch= str(epoch_index +1), duration_timer = 3000)  
                print(f"Avg loss for the epoch -> {loss_epoch/len(train_dataloader)}")
                loss_epochs.append(loss_epoch/len(train_dataloader))
                
                # save model periodically
                if (epoch_index+1)% freq_save == 0 and save_model:
                    saved = True
                    name_folder = os.path.join("./models",name_model+ "_" + date_)
                    name_ckpt =  str(epoch_index+1)
                    self._saveModel(name_ckpt, path_folder= name_folder)
                
                
                if early_stopping:    
                    # Data validation
                    acc = self.valid(valid_dataloader, epoch = epoch_index+1)
                    print(f"Accuracy from validation: {acc}")
                    valid_history.append(acc)
                    
                    # Early stopping
                    if epoch_index > 0:
                        if valid_history[-1] < valid_history[-2]:
                            if patience_counter >= self.patience:
                                print("Early stop")
                                break
                            else:
                                print("Pantience counter increased")
                                patience_counter += 1
                        else:
                            print("Accuracy increased respect previous epoch")
                

            # save last epoch if not already saved
            if not(saved):
                name_folder = os.path.join("./models",name_model+ "_" + date_)
                name_ckpt =  str(self.n_epochs)
                self._saveModel(name_ckpt, path_folder= name_folder)
            
            # print the plot with the avg loss over epochs
            name_folder = os.path.join("./results",name_model+ "_" + date_)
            self.plot_loss(loss_array= loss_epochs, epoch= str(self.n_epochs), path_save= name_folder ,duration_timer = 3000)
        
    
    def test(self, epoch_model, folder_model, verbose = True,):
        
        dataloader = self.dataset.get_testSet()
        n_steps = len(dataloader)
        name_model = folder_model.split('models/')[1]
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
        
        
    # ---------------- [Save & Load functions]
     
    def _saveModel(self, name, path_folder= "./models"):
        
        name_file = 'resNet3D-'+ str(name) +'.ckpt'
        path_save = os.path.join(path_folder, name_file)
        print("Saving model to: ", path_save)
        
        # create directories for models if doesn't exist
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
            
        T.save(self.model.state_dict(), path_save)
        
    def loadModel(self, epoch, path_folder= "./models"):
        
        name_file = 'resNet3D-'+ str(epoch) +'.ckpt'
        path_save = os.path.join(path_folder, name_file)
        print("Loading model from: ", path_save)
        ckpt = T.load(path_save)
        self.model.load_state_dict(ckpt)
    
    def saveJson(self, path, data):
        with open(path, "w") as file:
            json.dump(data, file, indent= 4)
        
    
    def loadJson(self, path):
        with open(path, "r") as file:
            json_data = file.read()
        data =  json.loads(json_data)
        return data
        
    # ---------------- [Monitoring model functions]
        
    def printSummaryNetwork(self, inputShape = (None,None,None,None)):
        summary(self.model, inputShape)

    
    def _computeMetrics(self,output, targets, name_model, epoch: str, labels = [0,1,2,3], average = "micro", save_results = True):  #labels not used
        
        if save_results: 
            # create path if doesn't exist
            path_save = os.path.join("./results", name_model)
            if not(os.path.exists(path_save)):
                os.makedirs(path_save)

        metrics_results = {
                            "precision":precision_score(targets, output, average = average, zero_division=1,),    \
                            "recall": recall_score(targets, output, average = average, zero_division=1),         \
                            "f1-score": f1_score(targets, output, average= average, zero_division=1),            \
                            # "average_precision": average_precision_score(targets, output, average= average),     \
                            "hamming_loss": hamming_loss(targets,output),                                        \
                            "jaccard_score": jaccard_score(targets, output, average= average, zero_division=1),  \
                            "confusion_matrix": confusion_matrix(targets,output, labels= [0,1,2,3], normalize= None)
        
            }
        
        if save_results:
            # save on json, use a copy to change numpy array with list
            metrics_results_ = metrics_results.copy()
            metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
            self.saveJson(os.path.join(path_save, 'testingMetrics_' + epoch + '.json'), metrics_results_)
            
            # np.save(os.path.join(path_save, "metrics.npy"),metrics_results)
        
        for k,v in metrics_results.items():
            if k != "confusion_matrix":
                print("\nmetric: {}, result: {}".format(k,v))
            else:
                print("Confusion matrix")
                # for kcm,vcm in v.items():
                #     print("\nconfusion matrix for: {}".format(kcm))
                print(v)
        
        self.plot_cm(cm = metrics_results['confusion_matrix'], path_save = path_save, epoch = epoch, duration_timer= None)
        
        return metrics_results
    
    def plot_cm(self,cm, path_save, epoch, duration_timer = 5000):
        
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
        
    def plot_loss(self, loss_array, epoch, path_save = None, duration_timer = 1000):
    
        def close_event():
            plt.close()
        
        # check if the folder exists otherwise create it
        if (path_save is not None) and (not os.path.exists(path_save)):
            os.makedirs(path_save)
        
        # plt.style.use('dark_background')
        
        # define x axis values
        x_values = list(range(1,len(loss_array)+1))
        
        # Generate an array of colors based on the values
        # colors = np.arange(len(loss_array))
        # plt.plot(x_values, loss_array, color = "red",linewidth=2)
        # plt.scatter(x_values, loss_array, c=colors, cmap='cool')
        
        # Create a colormap
        # cmap = colormaps.get_cmap('Reds')    
        # norm = Normalize(vmin= min(loss_array), vmax= max(loss_array))
        
        color = "green"
    
        # Plot the array with a continuous line color
        for i in range(len(loss_array) -1):
            # plt.plot([x_values[i], x_values[i + 1]], [loss_array[i], loss_array[i + 1]], color=cmap(norm(loss_array[i])) , linewidth=2)
            plt.plot([x_values[i], x_values[i + 1]], [loss_array[i], loss_array[i + 1]], color= color , linewidth=2)
            

        
        # text on the plot
        if path_save is None:       
            plt.xlabel('steps', fontsize=18)
        else:
            plt.xlabel('epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        
        plt.title('Learning loss plot', fontsize=18)
        
        # save if you define the path
        if path_save is not None:
            plt.savefig(os.path.join(path_save, 'loss_'+ epoch +'.png'))
        
        fig = plt.gcf()
        
        if duration_timer is not None:
            timer = fig.canvas.new_timer(interval=duration_timer)
            timer.add_callback(close_event)
            timer.start()
        
        plt.show()
    
        
    # ---------------- [Model forward function]
    def forward(self, x, verbose = False):
        """
            model forward, it first computes the logits, then the associated proababilities and return the most likely class.
            @ param x: input vector in batch or not, of the shape: (channels, frames, width , height)
        """
        
        # wrap for the batch
        if len(x.shape) < 5:
            x = np.expand_dims(x, 0)
        if len(x.shape) != 5:
            raise ValueError("Error in the input shape for the forward")
    
        # tranform in tensor if not 
        if not T.is_tensor(x):
            x = T.tensor(x)
        
        # correct the dtype
        if not (T.dtype is T.float32):
            x = x.to(T.float32)
        
        with T.no_grad():
            x = x.to(self.device)
            if verbose: print("input shape ->", x.shape) 
            
            # compute logits    
            logits  = self.model.forward(x)
            if verbose: print("logits shape ->",logits.shape) 
            
            # compute probabilities
            probs  = self.output_af(logits, dim = -1)               # -1, so on the last axis
            if verbose: print("probs shape ->",probs.shape)
            
            # get the class with max probability 
            y_pred = T.argmax(probs, dim= -1).cpu().detach().numpy().astype(int)
            if verbose: print("y_pred shape ->",y_pred.shape)

        
        return y_pred

# ----------------------------------------------------------------------------------------test functions 

def test_forward_2DNet():
    classifier = EngagementClassifier(args = None, video_resnet= False, depth_level = 0)

    # test summary model
    rand_input = T.rand(3,640,480).to(classifier.device)
    print(rand_input.shape)
    rand_input_batch = rand_input.unsqueeze(0)
    print(rand_input.shape)

    classifier.printSummaryNetwork(inputShape= rand_input.shape)
    x = classifier.model.forward(rand_input_batch)


def test_forward_3DNet():
    classifier = EngagementClassifier(args = None, video_resnet= True, depth_level = 0, grayscale= True)

    # test summary model
    rand_input = T.rand(1,30,640,480).to(classifier.device)
    print(rand_input.shape)
    rand_input_batch = rand_input.unsqueeze(0)
    print(rand_input_batch.shape)
    x = classifier.model.forward(rand_input_batch)
    print(x.shape)
    
    classifier.printSummaryNetwork(inputShape= rand_input.shape)
    
def test_training():
    classifier = EngagementClassifier(batch_size= 2, version_dataset= 'v2', grayscale= False)
    classifier.n_epochs = 2
    classifier.train(name_model= "test_1", save_model= True, verbose= False)
    # v1 -> {0: 4, 1: 81, 2: 861, 3: 777}
    # v2 -> {0: 4, 1: 19, 2: 107, 3: 106}
    
def test_validation():
    classifier = EngagementClassifier(batch_size= 2, version_dataset= 'v2', grayscale= False)
    classifier.loadModel(epoch = 1, path_folder="./models/test_valid_set_23-05-2023")
    avg_accuracy = classifier.valid(classifier.dataset.get_validationSet(), epoch=0)
    print(avg_accuracy)
    
def test_forward():
    classifier = EngagementClassifier()
    classifier.loadModel(epoch = 1, path_folder="./models/test_valid_set_23-05-2023")
    x = np.random.rand(2,3,30,480,640)
    y_pred = classifier.forward(x, verbose = True)
    print(y_pred, y_pred.shape)

def test_testing(name, epoch, version = 'v2', grayscale =False, depth_level = 0, batch_size = 1):
    classifier = EngagementClassifier(version_dataset= version, grayscale= grayscale, depth_level= depth_level, batch_size=batch_size)
    classifier.test(epoch_model= epoch, folder_model="./models/" + name, verbose= False)

def test_sampler():
    x = np.random.rand(1, 300, 3, 640, 480)
    classifier = EngagementClassifier()
    print(x.shape)
    for i in range(10):
        x_ = classifier.sample_frames(x=x, n_frames = 30, fps_sampling= 30, verbose = True)
        print(x_.shape, "\n")
        
def test_loss_plotting():
    classifier = EngagementClassifier(batch_size= 1, version_dataset= 'v2')
    loss = [2,10,40,100,25,13,6,3,2,1]
    classifier.plot_loss(loss, 30, duration_timer= None)


# test_forward_2DNet()
# test_forward_3DNet()
# test_training()
# test_validation()
# test_forward()
# test_sampler()
# test_loss_plotting()
# test_testing()

# ----------------------------------------------------------------------------------------training functions 

def train_1():
    classifier = EngagementClassifier(batch_size= 4, version_dataset= 'v2', grayscale= False)
    classifier.n_epochs = 10
    classifier.patience = 3
    classifier.train(name_model= "train_v2_batch4_color_depth0", save_model= True, verbose= False)
    
def train_2():
    classifier = EngagementClassifier(batch_size= 1, version_dataset= 'v2', grayscale= False)
    classifier.n_epochs = 15
    classifier.patience = 5
    classifier.train(name_model= "train_v2_batch1_color_depth0", save_model= True, verbose= False)  

def train_3():
    classifier = EngagementClassifier(batch_size= 4, version_dataset= 'v2', grayscale= False)
    classifier.n_epochs = 50
    classifier.patience = 10
    classifier.train(name_model= "train_v2_batch4_color_depth0_epochs50_patience10", save_model= True, verbose= False)

def train_4():
    classifier = EngagementClassifier(batch_size= 8, version_dataset= 'v2', grayscale= False)
    classifier.n_epochs = 50
    # classifier.patience = 50
    classifier.train(name_model= "train_v2_batch8_color_depth0_epochs50", save_model= True, verbose= False, early_stopping= False)

def train_5():
    classifier = EngagementClassifier(batch_size= 2, version_dataset= 'v3', grayscale= True)
    classifier.n_epochs = 5
    classifier.train(name_model= "train_v3_batch2_gray_depth0_epochs5", save_model= True, verbose= False)  
    # v3 time learning 33 minutes per epoch
    # v1 time learning 4h per epoch

# train_5()
# test_testing(name="train_v3_batch2_gray_depth0_epochs5_27-05-2023", epoch = 5, grayscale= True, batch_size= 2)







