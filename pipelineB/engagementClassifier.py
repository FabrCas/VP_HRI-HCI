import numpy as np
import os 
import random
random.seed(22)

# torch import
import torch as T
from torch import nn
from torchsummary import summary
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast

# custom classes
from models import ResNet2D, ResNet3D
from dataset import Dataset


class EngagementClassifier(nn.Module):
    def __init__(self, args, video_resnet = True, depth_level = 0):
        """
        @param args: list of arguments from main module 
        @param depth_level: int value to choose depth of the network
        
        """
        super().__init__()
        self.args = args
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.dataset: Dataset = Dataset()

        
        # build and load the model to GPU
        if video_resnet:
            self.model: ResNet2D = ResNet3D(depth_level = depth_level)
        else:
            self.model: ResNet2D = ResNet2D(depth_level = depth_level)
        self.model.to(self.device)
        
        # learning functions
        
        """ 
         softmax produces a probability distribution over classes, while log softmax transforms
         the softmax probabilities into a logarithmic scale for numerical stability during 
        """
        
        self.output_af  = nn.LogSoftmax(dim=1)
        self.loss_f     = nn.CrossEntropyLoss
        
        # learning parameters
        self.lr = 1e-4
        self.n_epochs = 5
        
    
        
    def train(self):
        dataloader = self.dataset.get_validationSet()  # TODO to change with train dataloader 
        
        n_steps = len(dataloader)
        print("number of steps per epoch: {}".format(n_steps))
        
        self.model.train()
        
        # define optimize , scheduler and scaler
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay= 0)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        scaler = GradScaler()
        
        # for n_epoch in range(self.n_epochs) :
        
        for step_index, (frames, label, timestamps, label_description) in enumerate(dataloader):
            x = self.sample_frames(frames)
            
            # expect input for the network: (batch_size, color_size, frames_len, width, height), so swap axis 2 and axis 1
            x = x.permute(0,2,1,3,4).to(self.device)  
            
            y = label
            
            # zeroing grad and cache
            T.cuda.empty_cache()    
            optimizer.zero_grad()
            
            x.requires_grad_()
            
            with autocast():
                logits  = self.model.forward(x)
                output  = self.output_af(logits)
                loss    = self.loss_f(output, y)
            
            # backpropagation
            scaler.scale(loss).backward()
            # clip gradients
            T.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # compute updates
            scaler.step(optimizer)
            # update weights
            scaler.update()
    
    def sample_frames(self, x, n_frames = 30):  # 30 frames corresponds to 1[s] of video
        # sample the frames taking a random number from 0 to (total_frames - n_frames) and include the next n_frames.
        total_frames = x.shape(1)
        print(total_frames)
        start_frame_index = random.randint(0,(total_frames - n_frames))
        end_frame_index = start_frame_index + n_frames
        x_sampled = x[:,start_frame_index:end_frame_index]
        return x
            
    def test(self):
        
        self.model.eval()
        pass
    
    
    def compute_class_weights(self, labels):
        class_freq={}
        total = len(labels)
        for l in labels:
            if l not in class_freq.keys():
                class_freq[l] = 0
            else:
                class_freq[l] = class_freq[l]+1
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(total/freq)

        return class_weights

    def printSummaryNetwork(self, inputShape = (None,None,None,None)):
        summary(self.model, inputShape)
        
    def _saveModel(self, epoch):
        path_save = os.path.join(self.args.saveDir)
        name = 'resNet-'+ str(epoch) +'.ckpt'
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        T.save(self.model.state_dict(), path_save + name)
        
    def loadModel(self, epoch = 70):
        path_load = os.path.join(self.args.saveDir + '/', 'resNet-{}.ckpt'.format(epoch))
        ckpt = T.load(path_load)
        self.model.load_state_dict(ckpt)
    
    def forward(self):
        pass



# test classifier 
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
    
# test_forward_2D()
# test_forward_3D()