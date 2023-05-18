import torch as T
import torch.nn as nn
import time

# ________________________________ ResNet _____________________________________

"""
                            Legend:
                                
    c_n -> 2D/3D convolutional layer number n
    bn_n -> 2D/3D batch normalization layer number n
    relu -> rectified linear unit activation function
    mp -> 2D/3D max pooling layer
    ds_layer -> downsampling layer
    ap -> average pooling layer
    fm_dim -> feature map dimension
    exp_coeff -> expansion coefficent
    
    
    depth_level:
    0 -> 18  layers ResNet
    1 -> 34  layers ResNet
    2 -> 50  layers ResNet
    3 -> 101 layers ResNet
    4 -> 152 layers ResNet
    
    
    input shape:
    2D -> [batch, colours, width, height]
    3D -> [batch, frames, colours, width, height]
"""
# ------------------------------------------------------ resnet blocks classes

class Bottleneck_block2D_l(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        super(Bottleneck_block2D_l,self).__init__()
        """
                            3 Convolutional layers
        """
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv2d(n_inCh, n_outCh, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv2d(n_outCh, n_outCh, kernel_size=3, stride=stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_outCh)
        
        # 3rd block
        self.c_3 = nn.Conv2d(n_outCh, n_outCh*self.exp_coeff, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(n_outCh*self.exp_coeff)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.relu(self.bn_2(self.c_2(x)))
        x = self.bn_3(self.c_3(x))
    
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
            
        # print(x.shape)
        # print(x_init.shape)
        
        x+=x_init
        x=self.relu(x)
        
        return x

class Bottleneck_block2D_s(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        """
                            2 Convolutional layers
        """
        super(Bottleneck_block2D_s,self).__init__()
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv2d(n_inCh, n_outCh, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv2d(n_outCh, n_outCh, kernel_size=3, stride = stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_outCh)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.bn_2(self.c_2(x))
        
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
            
        # print(x.shape)
        # print(x_init.shape)
    
        x+=x_init
        x=self.relu(x)
        
        return x
        
class Bottleneck_block3D_l(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        super(Bottleneck_block3D_l,self).__init__()
        """
                            3 Convolutional layers
        """
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv3d(n_inCh, n_outCh, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm3d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv3d(n_outCh, n_outCh, kernel_size=3, stride=stride, padding=1)
        self.bn_2 = nn.BatchNorm3d(n_outCh)
        
        # 3rd block
        self.c_3 = nn.Conv3d(n_outCh, n_outCh*self.exp_coeff, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm3d(n_outCh*self.exp_coeff)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.relu(self.bn_2(self.c_2(x)))
        x = self.bn_3(self.c_3(x))
    
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
            
        # print(x.shape)
        # print(x_init.shape)
        
        x+=x_init
        x=self.relu(x)
        
        return x

class Bottleneck_block3D_s(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        """
                            2 Convolutional layers
        """
        super(Bottleneck_block3D_s,self).__init__()
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv3d(n_inCh, n_outCh, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm3d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv3d(n_outCh, n_outCh, kernel_size=3, stride = stride, padding=1)
        self.bn_2 = nn.BatchNorm3d(n_outCh)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.bn_2(self.c_2(x))
        
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
    
        x+=x_init
        x=self.relu(x)
        
        return x

# ------------------------------------------------------ resnet models classes

class ResNet2D(nn.Module):
    # channel -> colors image
    # classes -> unique labels for the classification
    
    def __init__(self, depth_level = 2, n_channels = 3, n_classes = 4):
        super(ResNet2D,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.exp_coeff = 4 # output/input feature dimension ratio in bottleneck (default value for mid/big size model, reduced for small resnet 18 & 34)
    
        
        print("Creating the RNN (2D)...")
        self.initialTime = time.time()
        self.input_ch= 64   #    300 frames 64
        self.depth_level = depth_level
        self._check_depth_level()
        
        
        self.bottleneck_structs = [[2,2,2,2], [3,4,6,3], [3,4,6,3], [3,4,23,3], [3,8,36,3]]  # number of layers for the bottleneck                                             
        
        self.bottleneck_struct = self.bottleneck_structs[depth_level]
        self.fm_dim = [64,128,256,512]                          # feature map dimension
        
        
        self._create_net()
    
    def _check_depth_level(self):
        if self.depth_level< 0 or self.depth_level>4:
            raise ValueError("Wrong selection for the depth level, range: [0,4]")
        
        # set to 1 the exp coefficient if you are using reduced model
        if self.depth_level < 2: self.exp_coeff = 1
    
    
    def _create_net(self):
        # first block
        self.c_1 = nn.Conv2d(self.n_channels, self.fm_dim[0], kernel_size= 7, stride = 2, padding = 3, bias = False)
        self.bn_1 = nn.BatchNorm2d(self.fm_dim[0])
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size= 3, stride = 2, padding= 1)
        
        # body blocks
        self.l1 = self._buildLayers(n_blocks = self.bottleneck_struct[0], n_fm = self.fm_dim[0])
        self.l2 = self._buildLayers(n_blocks = self.bottleneck_struct[1], n_fm = self.fm_dim[1], stride = 2)
        self.l3 = self._buildLayers(n_blocks = self.bottleneck_struct[2], n_fm = self.fm_dim[2], stride = 2)
        self.l4 = self._buildLayers(n_blocks = self.bottleneck_struct[3], n_fm = self.fm_dim[3], stride = 2)
        
        
        # last block
        self.ap = nn.AdaptiveAvgPool2d((1,1)) # (1,1) output dimension
        self.fc = nn.Linear(512*self.exp_coeff, self.n_classes)
        
        # self.af_out = nn.Sigmoid() # used directly in the loss function
        
        print("Model created, time {} [s]".format(time.time() - self.initialTime))
        
    def _buildLayers(self, n_blocks, n_fm, stride = 1):
        
        """
            basic block 2 operations: conv + batch normalization
            bottleneck block 3 operations: conv + batch normalization + ReLU
        """
        list_layers = []
        
        # adapt x to be summed with output of the blocks
        if stride != 1 or self.input_ch != n_fm*self.exp_coeff: # so downsampling to handle the different shape of x 
            ds_layer = nn.Sequential(
                nn.Conv2d(self.input_ch, n_fm*self.exp_coeff, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_fm*self.exp_coeff)
                )
        else:
            ds_layer = None
    

        print("number of blocks fm {} -> {}".format(n_fm, n_blocks))
        # first layer to get the right feature map
        if self.depth_level < 2:
            print("building small block n°1")
            list_layers.append(Bottleneck_block2D_s(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        else:
            print("building big blocks n°1")
            list_layers.append(Bottleneck_block2D_l(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        self.input_ch = n_fm * self.exp_coeff
        
        # include all the layers from the bottleneck blocks
        for index, _ in enumerate(range(n_blocks -1)):
            if self.depth_level < 2:
                print("building small block n°{}".format(index +2))
                list_layers.append(Bottleneck_block2D_s(self.input_ch, n_fm))
            else:
                print("building big blocks n°{}".format(index +2))
                list_layers.append(Bottleneck_block2D_l(self.input_ch, n_fm))
        
        return nn.Sequential(*list_layers)
        
       
    def forward(self, x):
        # first block
        x = self.mp(self.relu(self.bn_1(self.c_1(x))))
        
        # body blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
    
        # last block
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

class ResNet3D(nn.Module):
    # channel -> colors image
    # classes -> unique labels for the classification
    
    def __init__(self, depth_level = 2, n_channels = 3, n_classes = 4):
        super(ResNet3D,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.exp_coeff = 4 # output/input feature dimension ratio in bottleneck   
    
        
        print("Creating the RNN (3D)...")
        self.initialTime = time.time()
        self.input_ch= 64   #    300 frames 64
        self.depth_level = depth_level
        self._check_depth_level()
        
        
        self.bottleneck_structs = [[2,2,2,2], [3,4,6,3], [3,4,6,3], [3,4,23,3], [3,8,36,3]]  # number of layers for the bottleneck                                             
        
        self.bottleneck_struct = self.bottleneck_structs[depth_level]
        self.fm_dim = [64,128,256,512]                          # feature map dimension
        
        
        self._create_net()
    
    def _check_depth_level(self):
        if self.depth_level< 0 or self.depth_level>4:
            raise ValueError("Wrong selection for the depth level, range: [0,4]")
        
        # set to 1 the exp coefficient if you are using reduced model
        if self.depth_level < 2: self.exp_coeff = 1
    
    
    def _create_net(self):
        # first block
        self.c_1 = nn.Conv3d(self.n_channels, self.fm_dim[0], kernel_size= 7, stride = 2, padding = 3, bias = False)
        self.bn_1 = nn.BatchNorm3d(self.fm_dim[0])
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool3d(kernel_size= 3, stride = 2, padding= 1)
        
        # body blocks
        self.l1 = self._buildLayers(n_blocks = self.bottleneck_struct[0], n_fm = self.fm_dim[0])
        self.l2 = self._buildLayers(n_blocks = self.bottleneck_struct[1], n_fm = self.fm_dim[1], stride = 2)
        self.l3 = self._buildLayers(n_blocks = self.bottleneck_struct[2], n_fm = self.fm_dim[2], stride = 2)
        self.l4 = self._buildLayers(n_blocks = self.bottleneck_struct[3], n_fm = self.fm_dim[3], stride = 2)
        
        
        # last block
        self.ap = nn.AdaptiveAvgPool3d((1,1,1)) # (1,1) output dimension
        self.fc = nn.Linear(512*self.exp_coeff, self.n_classes)
        
        # self.af_out = nn.Sigmoid() # used directly in the loss function
        
        print("Model created, time {} [s]".format(time.time() - self.initialTime))
        
    def _buildLayers(self, n_blocks, n_fm, stride = 1):
        
        """
            basic block 2 operations: conv + batch normalization
            bottleneck block 3 operations: conv + batch normalization + ReLU
        """
        list_layers = []
        
        # adapt x to be summed with output of the blocks
        if stride != 1 or self.input_ch != n_fm*self.exp_coeff: # so downsampling to handle the different shape of x 
            ds_layer = nn.Sequential(
                nn.Conv3d(self.input_ch, n_fm*self.exp_coeff, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_fm*self.exp_coeff)
                )
        else:
            ds_layer = None
    

        print("number of blocks fm {} -> {}".format(n_fm, n_blocks))
        # first layer to get the right feature map
        if self.depth_level < 2:
            print("building small block n°1")
            list_layers.append(Bottleneck_block3D_s(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        else:
            print("building big blocks n°1")
            list_layers.append(Bottleneck_block3D_l(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        self.input_ch = n_fm * self.exp_coeff
        
        # include all the layers from the bottleneck blocks
        for index, _ in enumerate(range(n_blocks -1)):
            if self.depth_level < 2:
                print("building small block n°{}".format(index +2))
                list_layers.append(Bottleneck_block3D_s(self.input_ch, n_fm))
            else:
                print("building big blocks n°{}".format(index +2))
                list_layers.append(Bottleneck_block3D_l(self.input_ch, n_fm))
        
        return nn.Sequential(*list_layers)
        
       
    def forward(self, x):
        # first block
        x = self.mp(self.relu(self.bn_1(self.c_1(x))))
        
        # body blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
    
        # last block
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x