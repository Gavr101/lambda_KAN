import os
import sys
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn

from kan import KANLayer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))
from source import lmdKANLayer, tlmdKANLayer

import random


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(42)


class MnistLR(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        '''
        Linear Regression model. nn.Linear layer used for sensitivity illustration after fitting
        '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
    def get_W(self):
        '''
        Function for online sensitivity logging.
        '''
        LR_layer = self.layers[1].state_dict()
        W = LR_layer['weight'].permute(1,0).detach().clone().cpu().numpy()
        
        return W


# ======

class MnistMLP(nn.Module):
    '''
    MLP model, as reference.
    '''
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.ReLU
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            self.act(),
            nn.Linear(256, 64),
            self.act(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# ======

class lmdSplineKANLayer(lmdKANLayer):
    '''
    Filters output from self.forward().
    '''
    def forward(self, x):
        y, preacts, postacts, postspline, beforelmd = super().forward(x)
        return y
    

class Mnist_lmdSplineKAN(nn.Module):
    '''
    Main explainable model. 
    '''
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.Tanh
        self.l_layers = nn.ModuleList()
        for i in range(10):
            layers = nn.Sequential(
                nn.Flatten(),
                lmdSplineKANLayer(in_dim=28 * 28, out_dim=64, 
                                  num=5, k=3, 
                                  grid_range=[0, 1]),
                self.act(),
                nn.Linear(64, 32),
                self.act(),
                nn.Linear(32, 1),
            )
            self.l_layers.append(layers)

    def forward(self, x):
        l_logits = []
        for i in range(10):
            l_logits.append(self.l_layers[i](x))
            
        y = torch.cat(l_logits, dim=-1)
        return y
    
    def get_W(self):
        '''
        Function for online sensitivity logging.
        '''
        lmd_W = []
        for i in range(10):
            lmd_W.append(self.l_layers[i][1].lmd.detach().clone().cpu().numpy())

        lmd_W = np.array(lmd_W).transpose(1,0)
        
        return lmd_W
    
    
# ======

class tlmdSplineKANLayer(tlmdKANLayer):
    '''
    Filters output from self.forward().
    '''
    def forward(self, x):
        y, preacts, postacts, postspline, beforelmd, add_lmd_copy = super().forward(x)
        return y
    

class Mnist_tlmdSplineKAN(nn.Module):
    '''
    Main explainable model. 
    '''
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.Tanh
        self.l_layers = nn.ModuleList()
        for i in range(10):
            layers = nn.Sequential(
                nn.Flatten(),
                tlmdSplineKANLayer(in_dim=28 * 28, out_dim=64, 
                                  num=5, k=3, 
                                  grid_range=[0, 1],
                                  tlmd_alpha=1e-1),
                self.act(),
                nn.Linear(64, 32),
                self.act(),
                nn.Linear(32, 1),
            )
            self.l_layers.append(layers)

    def forward(self, x):
        l_logits = []
        for i in range(10):
            l_logits.append(self.l_layers[i](x))
            
        y = torch.cat(l_logits, dim=-1)
        return y
    
    def get_W(self):
        '''
        Function for online sensitivity logging.
        '''
        lmd_W = []
        for i in range(10):
            lmd_W.append(self.l_layers[i][1].lmd.detach().clone().cpu().numpy())

        lmd_W = np.array(lmd_W).transpose(1,0)
        
        return lmd_W


# ======

def show_sensitivity(W, title = '', save_pic=False, file_path=None):
    '''
    Shows sensitivity maps for Linear Regression and Lambda KAN models.
    
    W : np.array
        size = [28*28, 10]
    '''
    # Display templates
    plt.rcParams["figure.figsize"] = (25, 10)

    #W = np.loadtxt("lc_mnist_weights.txt")  # load weigths, shape (785, 10)
    print(f"Shape with bias: {W.shape}")

    ## Remove bias
    #W = W[:-1, :]
    #print(f"Shape without bias: {W.shape}")

    # Normalize
    w_min, w_max = np.min(W), np.max(W)
    templates = 255 * (W - w_min) / (w_max - w_min)

    # Display templates
    labels_names = [str(i) for i in range(10)]
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        img = templates[:, i].reshape(28, 28).astype(int)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(labels_names[i], size=25)
        
    plt.suptitle(title, size=50)
    plt.tight_layout()
    plt.subplots_adjust(top=1.44)
    if save_pic:
        plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
# ======

class MnistCNN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # [1, 28, 28]
            nn.Conv2d(1, 5, 3, padding=1),  # in channel=1, out=5
            nn.ReLU(),
            # [5, 28, 28]
            nn.MaxPool2d(2),
            # [5, 14, 14]
            nn.Conv2d(5, 10, 3, padding=1),  # in channel=5, out=10
            nn.ReLU(),
            # [10, 14, 14]
            nn.MaxPool2d(2),
            # [10, 7, 7]
        )
        self.cl = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, 100),  # in = channel*heght*width
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.cl(x)
        return x
    
    
# ======

class ReLUKANLayer(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 grid: int, 
                 overlap : int = 1,  
                 train_ab: bool = True,
                 basis_f_left_boarder : float = -1.,
                 basis_f_right_boarder : float = 1.,
                 init_mu : float = 0.,
                 init_sigma : float = 1.):
        '''
        KAN layer with relu-based activation functions basis.
        
        Args:
        -----
            input_size : int
                Size of input vector.
            output_size : 
                Size of output vector.
            grid : int
                Number of basis functions per each learning 1-variable function.
            overlap : float>0
                Measure of overlapping initialized basis functions. 0 makes basis f. as delta-functions; (0,1] - central basis f. overlapping with 2 others basis.f.. 
            train_ab : bool
                Whether trainable basis functions or not.
            basis_f_left_boarder : float
                Left boarder of initialized basis functions` grid. Corresponds to the leftmost basis f. center position.
            basis_f_right_boarder : float
                Right boarder of initialized basis functions` grid. Corresponds to the rightmost basis f. center position.
            init_mu : float
                Mean of initialisation distribution of coefficients.
            init_sigma : float
                Dispertion of initialisation distribution of coefficients.
        '''
        super().__init__()
        self.grid, self.overlap = grid, overlap
        self.input_size, self.output_size = input_size, output_size
        self.basis_f_left_boarder, self.basis_f_right_boarder = basis_f_left_boarder, basis_f_right_boarder
        
        basis_f_centers = np.linspace(basis_f_left_boarder, basis_f_right_boarder, self.grid) # centers of basis. functions
        basis_f_step = (basis_f_right_boarder - basis_f_left_boarder) / (self.grid - 1)
        phase_low = basis_f_centers - basis_f_step
        phase_height = basis_f_centers + basis_f_step
        
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        
        #self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
        
        # torch.Size([input_size, output_size, grid])
        self.coff = torch.nn.Parameter(init_mu + init_sigma * (torch.randn(self.input_size, self.output_size, self.grid))).requires_grad_(True)
        
        
    def forward(self, x):
        # input x:              torch.Size([Batch, input_size, 1])
        # self.phase_low:       torch.Size([input_size, grid])
        # self.phase_height:    torch.Size([input_size, grid])
        
        # Norming coefficient norms each basis function to [0,1] range. Don`t provide gradients.
        norm_coeff = (self.phase_height.detach().clone() - self.phase_low.detach().clone())**4 / 16
        
        x = x.unsqueeze(dim=-1) # torch.Size([Batch, input_size, 1])
        #print(f'x:\t\t\t {x.shape}')
        x1 = torch.relu(x - self.phase_low) # torch.Size([Batch, input_size, grid])
        #print(f'x1:\t\t\t {x1.shape}')
        x2 = torch.relu(self.phase_height - x) # torch.Size([Batch, input_size, grid])
        #print(f'x2:\t\t\t {x2.shape}')
        x = x1 * x2 # torch.Size([Batch, input_size, grid])
        #print(f'x1 * x2:\t\t {x2.shape}')
        x = x**2 * norm_coeff # torch.Size([Batch, input_size, grid])
        #print(f'x**2:\t\t\t {x.shape}')
        x = x.unsqueeze(dim=2) # torch.Size([Batch, input_size, 1, grid])
        x = (self.coff * x).sum(dim=(1,3), keepdim=False) # torch.Size([Batch, output_size])
        #print(f'reshape x:\t\t {x.shape}')
        return x
    
    
class MnistReLUKAN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.Tanh
        self.layers = nn.Sequential(
            nn.Flatten(),
            ReLUKANLayer(input_size=28 * 28, output_size=10, 
                         grid=10, overlap=1.,
                         basis_f_left_boarder=0, basis_f_right_boarder=1,
                         init_mu=1),
            #self.act(),
            #nn.Linear(32, 16),
            #self.act(),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
    
# ======    

class SplineKANLayer(KANLayer):
    def forward(self, x):
        y, preacts, postacts, postspline = super().forward(x)
        return y
    
    
    
class MnistSplineKAN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.Tanh
        self.layers = nn.Sequential(
            nn.Flatten(),
            SplineKANLayer(in_dim=28 * 28, out_dim=10, 
                           num=5, k=3),
            #self.act(),
            #nn.Linear(64, 32),
            #self.act(),
            nn.Linear(10, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
    
# ======

class MnistPrllSplineKAN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.act = nn.Tanh
        self.l_layers = nn.ModuleList()
        for i in range(10):
            layers = nn.Sequential(
                nn.Flatten(),
                SplineKANLayer(in_dim=28 * 28, out_dim=16, 
                               num=5, k=3),
                self.act(),
                nn.Linear(16, 8),
                self.act(),
                nn.Linear(8, 1),
            )
            self.l_layers.append(layers)

    def forward(self, x):
        l_logits = []
        for i in range(10):
            l_logits.append(self.l_layers[i](x))
            
        y = torch.cat(l_logits, dim=-1)
        return y
    
    
# ======