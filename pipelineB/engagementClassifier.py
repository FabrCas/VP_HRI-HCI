import torch as T
from torchsummary import summary
import numpy as np
import os 


class EngagementClassifier(T.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        
    def train():
        pass
    
    def test():
        pass
    
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
    
    def forward():
        pass

    