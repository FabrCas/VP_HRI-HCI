import numpy as np
import os 
import random
import warnings
import gc

from datetime import date
from tqdm import tqdm
random.seed(22)

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
    def __init__(self, args = None, load = False, video_resnet = True, depth_level = 0, batch_size = 1):
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
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.dataset: Dataset = Dataset(batch_size = self.batch_size)

        # build and load the model to GPU
        if video_resnet:
            self.model: ResNet2D = ResNet3D(depth_level = depth_level)
        else:
            self.model: ResNet2D = ResNet2D(depth_level = depth_level)
        self.model.to(self.device)
        if not(load):
            self.init_weights_normal()
        else: 
            pass
        
        # define learning functions
        """ 
         softmax produces a probability distribution over classes, while log softmax transforms
         the softmax probabilities into a logarithmic scale for numerical stability during 
        """
        self.output_af  = F.softmax         # F.log_softmax
        self.loss_f     = F.cross_entropy   #nn.CrossEntropyLoss
        
        # learning parameters
        self.lr = 1e-4
        self.n_epochs = 1 # 5
        self.learning_weights = [161.235, 25.617, 2.069, 2.121]   # weights from self.compute_class_weights(), no needed to recompute
    
    # ---------------- [initialization functions]
    
    def init_weights_normal(self):
        print("Initializing weights using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.normal_(param, mean=0, std=0.01) 
                
    def init_weights_kaimingNormal(self):
        # Initialize the weights  using He initialization
        print("Initializing weights using He initialization")
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    # ---------------- [data processing functions]
    
    def sample_frames(self, x, n_frames = 30, verbose = False):  # 30 frames corresponds to 1[s] of video
        # avoid last frame, sometimes complete black frame
        # sample the frames taking a random number from 0 to ((total_frames -1) - n_frames) and include the next n_frames.
        
        #TODO pass to a 15 fps sampling (take a sample and skip the next one), in this case 2 seconds of recording are taken
        # probably it's better to use directlu in the dataset class
        
        total_frames = (x.shape[1] -1)
        start_frame_index = random.randint(0,(total_frames - n_frames))
        end_frame_index = start_frame_index + n_frames
        if verbose: print(f"start: {start_frame_index}, end:{end_frame_index}")
        x_sampled = x[:,start_frame_index:end_frame_index]
        return x_sampled
    
    def compute_class_weights(self):
        
        print("Computing the weights for the training set")
        # get train dataset using single mini-batch size
        laoder = self.dataset.get_dataloaderLabelsTrain()
        
        # compute occurrences of labels
        class_freq={}
        total = len(laoder)
        for y  in tqdm(laoder, total= total):
            l = y.item()
            if l not in class_freq.keys():
                class_freq[l] = 1
            else:
                class_freq[l] = class_freq[l]+1
        print("class_freq -> ", class_freq)
        
        # compute the weights   
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(total/freq)

        print("class_weights-> ", class_weights)
        return class_weights
    
    
    def focal_loss(y_pred, y_true, alpha=None, gamma=2, reduction='mean'):
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
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            # Apply class-specific alpha weights
            alpha = alpha.to(y_pred.device)
            focal_loss = alpha * focal_loss

        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss
    
    # ---------------- [Train & Test functions]

    def train(self, name_model = "test", save_model = False, verbose = True):
        dataloader = self.dataset.get_validationSet() # TODO to change with train dataloader 
        
        # get number of steps for epoch and current date in format "DD-MM-YYYY"
        n_steps = len(dataloader)
        date_ = date.today().strftime("%d-%m-%Y")
        print("number of steps per epoch: {}".format(n_steps))
        
        self.model.train()
        
        # define optimizer , scheduler and scaler
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay= 0)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        scaler = GradScaler()
        
        # initialize list to store the losses of each epoch
        loss_epochs = []
        
        # boolean flag indicing if the corrent epoch model has been savec
        saved: bool = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            for epoch_index in range(self.n_epochs) :
                loss_epoch = 0
                saved = False
                
                # cumulative loss for print
                tmp_loss = 0
                
                for step_index, (frames, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
                    
                    # 1) get x and y data
                    x = self.sample_frames(frames)
                    # TODO extract face
                    del frames; gc.collect()
                    # expect input for the network: (batch_size, color_size, frames_len, width, height), so swap axis 2 and axis 1
                    x = x.permute(0,2,1,3,4).to(self.device)  
                    x.requires_grad_()
                    
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
                        
                        loss = self.loss_f(logits, y, reduction= 'sum')
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
                    if step_index % 50 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                                .format(epoch_index, self.n_epochs, str(step_index).zfill(3), str(n_steps-1).zfill(3),
                                        scheduler.get_last_lr()[0], \
                                        tmp_loss))
                        tmp_loss = 0  # resetting the periodic loss for steps
                        
                print(f"Cumulative loss for the epoch -> {loss_epoch}")
                loss_epochs.append(loss_epoch)
                loss_epoch = 0
                
                # save model periodically
                if (epoch_index+1)%5 == 0 and save_model:
                    saved = True
                    name_folder = os.path.join("./models",name_model+ "_" + date_)
                    name_ckpt =  str(epoch_index+1)
                    self._saveModel(name_ckpt, path_folder= name_folder)

            # save last epoch if not already saved
            if not(saved):
                name_folder = os.path.join("./models",name_model+ "_" + date_)
                name_ckpt =  str(self.n_epochs)
                self._saveModel(name_ckpt, path_folder= name_folder)
        
    
    def test(self):
        self.model.eval()
        
    # ---------------- [Save & Load model functions]
     
    def _saveModel(self, name, path_folder= "./models"):
        name_file = 'resNet3D-'+ str(name) +'.ckpt'
        path_save = os.path.join(path_folder, name_file)
        print("Saving model to: ", path_save)
        
        # create directories for models if doesn't exist
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        T.save(self.model.state_dict(), path_save)
        
    def loadModel(self, name, path_folder= "./models"):
        
        name_file = 'resNet3D-'+ str(name) +'.ckpt'
        path_save = os.path.join(path_folder, name_file)
        print("Loading model from: ", path_save)
        ckpt = T.load(path_save)
        self.model.load_state_dict(ckpt)
    
    # ---------------- [Monitoring model functions]
        
    def printSummaryNetwork(self, inputShape = (None,None,None,None)):
        summary(self.model, inputShape)
    
    
    # ---------------- [Model forward function]
    def forward(self):
        pass



# ----------------------------------------------------------------------------------------test classifier 

def test_forward_2D():
    classifier = EngagementClassifier(args = None, video_resnet= False, depth_level = 0)

    # test summary model
    rand_input = T.rand(3,640,480).to(classifier.device)
    print(rand_input.shape)
    rand_input_batch = rand_input.unsqueeze(0)
    print(rand_input.shape)

    classifier.printSummaryNetwork(inputShape= rand_input.shape)
    x = classifier.model.forward(rand_input_batch)


def test_forward_3D():
    classifier = EngagementClassifier(args = None, video_resnet= True, depth_level = 0)

    # test summary model
    rand_input = T.rand(300,3,640,480).to(classifier.device)
    print(rand_input.shape)
    rand_input_batch = rand_input.unsqueeze(0)
    print(rand_input.shape)

    # classifier.printSummaryNetwork(inputShape= rand_input.shape)
    x = classifier.model.forward(rand_input_batch)
    
def test_training():
    classifier = EngagementClassifier(batch_size= 1)
    classifier.n_epochs = 1
    classifier.train(name_model= "test_valid_set", save_model= True, verbose= False)
    
# test_forward_2D()
# test_forward_3D()
# test_training()


classifier = EngagementClassifier(batch_size= 1)
classifier.compute_class_weights()