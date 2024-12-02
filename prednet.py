'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# As in the paper, the first layer is a regular conv layer (to reduce memory consumption)
class features2(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=7, stride=2, padding=3, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.featBN = nn.BatchNorm2d(outchan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.featBN(self.conv(x)))
        return y

class PcConvBp(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, cls=0, bias=False):
        super().__init__()
        self.FFconv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, kernel_size, stride, padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,outchan,1,1))])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.cls = cls
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x, layer_idx):
        y = self.relu(self.FFconv(x))
        for _ in range(self.cls):
            y = self.FFconv(self.relu(x - self.FBconv(y))) + y
        y = y + self.bypass(x)
        return y

    def la_sigmoid(self, x):
        return 0.5+0.25*x-0.0212*x**3

''' Architecture PredNetBpD_5 '''
class PredNetBpD_5(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False):
        super().__init__()
        self.ics = [3,  32, 64, 64, 128] # input chanels
        self.ocs = [32, 64, 64, 128, 128] # output chanels
        self.maxpool = [False, True, False, True, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


''' Architecture PredNetBpD '''
class PredNetBpD_3(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False):
        super().__init__()
        self.ics = [3,  32, 64] # input chanels
        self.ocs = [32, 64, 64] # output chanels
        self.maxpool = [False, True, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x, i)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class PcConvBp_SGD(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=3, stride=1, padding=1, cls=0, bias=False):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.FFconv = nn.Conv2d(inchan, outchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.FBconv = nn.ConvTranspose2d(outchan, inchan, self.kernel_size, self.stride, self.padding, bias=bias)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, outchan, 1, 1))])
        self.relu = nn.ReLU(inplace=True)
        self.cls = cls
        self.bypass = nn.Conv2d(inchan, outchan, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        y = self.relu(self.FFconv(x))
        y, energies = self.find_optimal_r(x, y)
        y = y + self.bypass(x)
        return y

    def find_optimal_r(self, x, y):
        weight = self.FBconv.weight.data
        expanded_weights = self.expand_weights_to_matrix(y.shape[1:], weight.permute(1, 0, 2, 3), stride=self.stride, padding=self.padding)
        expanded_weights = expanded_weights.to(y.device)
        flattened_x = x.view(1, -1).clone().detach()

        """
        Implement with SGD
        """
        # Initialize flattened_y as a tensor with requires_grad=True
        num_iterations = 1
        y = F.pad(y, (self.padding, self.padding, self.padding, self.padding))
        flattened_y = y.view(1, -1).clone().detach().requires_grad_(True)
        energy_record = []
        optimizer = torch.optim.SGD([flattened_y], lr=0.01)
        for _ in range(num_iterations):
            optimizer.zero_grad()
            energy = self.Energy_Function(flattened_x, expanded_weights, flattened_y)
            energy.backward()
            optimizer.step()
            energy_record.append(energy.item())

        # Reshape the flattened_y to the original shape
        _, C_in, H_in, W_in = y.shape
        C_out, _, K, _ = weight.shape
        H_out = (H_in - K + 2 * self.padding) // self.stride + 1
        W_out = (W_in - K + 2 * self.padding) // self.stride + 1
        optimal_y = flattened_y.view(-1, C_out, H_out, W_out)
        # Cut off the padding area
        optimal_y = optimal_y[:, :, self.padding:-self.padding, self.padding:-self.padding]
        optimal_y = optimal_y.to(y.device)

        return optimal_y.detach(), energy_record

    def Energy_Function(self, x, W, y):
        energy = -2* x @ W @ y.T + y @ (W.T @ W) @ y.T
        return energy
    
    def expand_weights_to_matrix(self, input_shape, weight_tensor, stride=1, padding=0):
        """
        Expand the convolution weights to a matrix suitable for multiplying with a flattened input vector.
        
        Args:
        - input_shape (tuple): Shape of the input (C_in, H_in, W_in)
        - weight_tensor (torch.Tensor): Convolution weights of shape (C_out, C_in, K, K)
        - stride (int): Stride of the convolution
        - padding (int): Padding size

        Returns:
        - expanded_weights (torch.Tensor): The expanded weight matrix for matrix multiplication
        """
        C_in, H_in, W_in = input_shape
        C_out, _, K, _ = weight_tensor.shape
        
        # Compute output dimensions
        H_out = (H_in + 2 * padding - K) // stride + 1
        W_out = (W_in + 2 * padding - K) // stride + 1

        # Initialize expanded weight matrix
        expanded_weights = torch.zeros((C_out * H_out * W_out, C_in * (H_in + 2 * padding) * (W_in + 2 * padding)))

        # Fill the expanded weight matrix
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # Calculate the starting index for each filter application
                    start_h = h * stride
                    start_w = w * stride
                    # Flattened receptive field index
                    filter_idx = c_out * H_out * W_out + h * W_out + w
                    # Fill the appropriate section of the expanded weight matrix
                    for c_in in range(C_in):
                        for i in range(K):
                            for j in range(K):
                                # Calculate the input index considering padding
                                input_idx = (c_in * (H_in + 2 * padding) + (start_h + i)) * (W_in + 2 * padding) + (start_w + j)
                                # Assign the weight to the correct position
                                expanded_weights[filter_idx, input_idx] = weight_tensor[c_out, c_in, i, j]

        return expanded_weights


''' Architecture PredNetBpE '''
class PredNetBpE(nn.Module):
    def __init__(self, num_classes=1000, cls=0, Tied = False):
        super().__init__()
        self.ics =     [    3,   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512] # input chanels
        self.ocs =     [   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512,  512] # output chanels
        self.maxpool = [False,False, True,False, True,False, True,False,False, True,False,False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        self.baseconv = features2(self.ics[0], self.ocs[0])
        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(1, self.nlays)])
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(1, self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(1, self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        x = self.baseconv(x)
        for i in range(self.nlays-1):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = self.relu(self.BNend(x))
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

''' Architecture PredNetBpD '''
class PredNetBpD(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False, solver=None):
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 512] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 512, 512] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            if solver is None:
                print('No solver in used, still using convolution in recurrent layer')
                self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
            elif solver == 'SGD':
                print(f'Solver {solver} is in use')
                self.PcConvs = nn.ModuleList([PcConvBp_SGD(3, 64, cls=self.cls)])
                self.PcConvs.extend([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(1, self.nlays)])
            else:
                print(f'Solver {solver} not supported')
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])

        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

''' Architecture PredNetBpC '''
class PredNetBpC(nn.Module):
    def __init__(self, num_classes=10, cls=0, Tied = False):
        super().__init__()
        self.ics = [3,  64, 64, 128, 128, 256, 256, 256] # input chanels
        self.ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        self.maxpool = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of time steps
        self.nlays = len(self.ics)

        # construct PC layers
        # Unlike PCN v1, we do not have a tied version here. We may or may not incorporate a tied version in the future.
        if Tied == False:
            self.PcConvs = nn.ModuleList([PcConvBp(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        else:
            self.PcConvs = nn.ModuleList([PcConvBpTied(self.ics[i], self.ocs[i], cls=self.cls) for i in range(self.nlays)])
        self.BNs = nn.ModuleList([nn.BatchNorm2d(self.ics[i]) for i in range(self.nlays)])
        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.BNend = nn.BatchNorm2d(self.ocs[-1])

    def forward(self, x):
        for i in range(self.nlays):
            x = self.BNs[i](x)
            x = self.PcConvs[i](x)  # ReLU + Conv
            if self.maxpool[i]:
                x = self.maxpool2d(x)

        # classifier                
        out = F.avg_pool2d(self.relu(self.BNend(x)), x.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

