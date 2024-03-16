"""Implements common unimodal encoders."""
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models as tmodels




class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""
    
    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.
        
        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        
        """
        return self.fc(x)


class Squeeze(torch.nn.Module):
    """Custom squeeze module for easier Sequential usage."""
    
    def __init__(self, dim=None):
        """Initialize Squeeze Module.

        Args:
            dim (int, optional): Dimension to Squeeze on. Defaults to None.
        """ 
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Apply Squeeze Layer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)


class Sequential(nn.Sequential):
    """Custom Sequential module for easier usage."""
    
    def __init__(self, *args, **kwargs):
        """Initialize Sequential Layer."""
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Apply args to Sequential Layer."""
        if 'training' in kwargs:
            del kwargs['training']
        return super().forward(*args, **kwargs)


class Reshape(nn.Module):
    """Custom reshape module for easier Sequential usage."""
    
    def __init__(self, shape):
        """Initialize Reshape Module.

        Args:
            shape (tuple): Tuple to reshape input to
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """Apply Reshape Module to Input.

        Args:
            x (torch.Tensor): Layer Input 

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.reshape(x, self.shape)


class Transpose(nn.Module):
    """Custom transpose module for easier Sequential usage."""
    def __init__(self, dim0, dim1):
        """Initialize Transpose Module.

        Args:
            dim0 (int): Dimension 1 of Torch.Transpose
            dim1 (int): Dimension 2 of Torch.Transpose
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        """Apply Transpose Module to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.transpose(x, self.dim0, self.dim1)


class MLP(torch.nn.Module):
    """Two layered perceptron."""
    
    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2



class LeNet(nn.Module):
    """Implements LeNet.
    
    Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """
    
    def __init__(self, in_channels, args_channels, additional_layers, output_each_layer=False, linear=None, squeeze_output=True):
        """Initialize LeNet.

        Args:
            in_channels (int): Input channel number.
            args_channels (int): Output channel number for block.
            additional_layers (int): Number of additional blocks for LeNet.
            output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
            linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
            squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.
        """
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.convs = [
            nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        self.bns = [nn.BatchNorm2d(args_channels)]
        self.gps = [GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(nn.Conv2d((2**i)*args_channels, (2**(i+1))
                              * args_channels, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(args_channels*(2**(i+1))))
            self.gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.gps = nn.ModuleList(self.gps)
        self.sq_out = squeeze_output
        self.linear = None
        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        """Apply LeNet to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        tempouts = []
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
            gp = self.gps[i](out)
            tempouts.append(gp)
            
        if self.linear is not None:
            out = self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out



class GlobalPooling2D(nn.Module):
    """Implements 2D Global Pooling."""
    
    def __init__(self):
        """Initializes GlobalPooling2D Module."""
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        """Apply 2D Global Pooling to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x


class Constant(nn.Module):
    """Implements a module that returns a constant no matter the input."""
    
    def __init__(self, out_dim):
        """Initialize Constant Module.

        Args:
            out_dim (int): Output Dimension.
        """
        super(Constant, self).__init__()
        self.out_dim = out_dim

    def forward(self, x):
        """Apply Constant to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.zeros(self.out_dim).to(x.device)


