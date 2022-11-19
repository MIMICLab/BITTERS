import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Downsampler(nn.Module):
    def __init__(self, z_channels):
        super(Downsampler, self).__init__()  
        self.downsample_block = nn.Sequential(
                                    nn.BatchNorm2d(z_channels),
                                    nn.PReLU(z_channels),             
                                    nn.Conv2d(z_channels, z_channels, 3, 2, 1),                                                                   
                                )
    def forward(self, x):
        out = self.downsample_block(x)
        return out

class Upsampler(nn.Module):
    def __init__(self, z_channels, upscale=2):
        super(Upsampler, self).__init__()  
        self.upsample_block = nn.Sequential(
                                    nn.BatchNorm2d(z_channels),
                                    nn.PReLU(z_channels),             
                                    nn.Conv2d(z_channels, z_channels * upscale * upscale, 3, 1, 1),         
                                    nn.PixelShuffle(upscale)                        
                                )
    def forward(self, x):
        out = self.upsample_block(x)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super(ResnetBlock, self).__init__() 
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.conv1 = nn.Sequential(
                                   nn.BatchNorm2d(in_channels),
                                   nn.PReLU(in_channels),
                                   nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                                   )
        self.conv2 = nn.Sequential(
                                   nn.BatchNorm2d(in_channels),
                                   nn.PReLU(in_channels),
                                   nn.Conv2d(in_channels, out_channels, 3, 1, 1)                            
                                   )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class ResBlocks(nn.Module):
    def __init__(self, hidden_dim, num_res_blocks=16):
        super(ResBlocks, self).__init__() 
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        blocks = []
        for _ in range(self.num_res_blocks):
            blocks.append(ResnetBlock(in_channels=hidden_dim,
                                    out_channels=hidden_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        h = z 
        out = self.blocks(h)
        return out

class WaveEncoder(nn.Module):
    def __init__(self, hidden_dim, in_channels=3, out_channels=128, num_res_blocks=16):
        super(WaveEncoder, self).__init__() 
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # 256 x 256 -> 128 x 128
        self.conv_in_1 = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1)
        self.body_1 = ResBlocks(hidden_dim, num_res_blocks)
        self.downsample_1 = Downsampler(hidden_dim)                                                
        self.conv_out_1 = nn.Conv2d(hidden_dim, out_channels, 1, 1)    

        # 128 x 128 -> 64 x 64
        self.conv_in_2 = nn.Conv2d(in_channels, hidden_dim * 2, 3, 1, 1)
        self.body_2 = ResBlocks(hidden_dim * 2, num_res_blocks)
        self.downsample_2 = Downsampler(hidden_dim * 2)                                     
        self.conv_out_2 = nn.Conv2d(hidden_dim * 2, out_channels, 1, 1)

        # 64 x 64 -> 32 x 32
        self.conv_in_3 = nn.Conv2d(in_channels, hidden_dim * 4, 3, 1, 1)
        self.body_3 = ResBlocks(hidden_dim * 4, num_res_blocks)   
        self.downsample_3 = Downsampler(hidden_dim * 4)                                                
        self.conv_out_3 = nn.Conv2d(hidden_dim * 4, out_channels, 1, 1)     

        # 32 x 32 -> 16 x 16
        self.conv_in_4 = nn.Conv2d(in_channels, hidden_dim * 8, 3, 1, 1)
        self.body_4 = ResBlocks(hidden_dim * 8, num_res_blocks)   
        self.downsample_4 = Downsampler(hidden_dim * 8)                                                
        self.conv_out_4 = nn.Conv2d(hidden_dim * 8, out_channels, 1, 1)             

        self.auxiliary = True   

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False  
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def setup_finetune(self):
        self.auxiliary = False
        del self.conv_out_1
        del self.conv_out_2
        del self.conv_in_2
        del self.conv_in_3
        self.conv_in_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 1, 1)
        self.conv_in_3 = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, 1, 1)

        del self.conv_in_4
        del self.body_4
        del self.downsample_4
        del self.conv_out_4

    def forward(self, x, x_high=None, x_mid=None, x_low=None):
        # feature extraction
        x_1 = x
        h_1 = self.conv_in_1(x_1)
        # body 1
        in_1 = h_1
        h_1 = self.body_1(h_1)
        h_1 = self.downsample_1(h_1)

        # residual 1
        res_1 = h_1 + F.avg_pool2d(in_1, 2)          
        if self.auxiliary:
            out_1 = self.conv_out_1(res_1)
            if x_high != None:
                x_2 = x_high
            else:
                x_2 = torch.clamp(out_1.detach(), -1.0, 1.0)
            h_2 = self.conv_in_2(x_2)   
        else:
            h_2 = self.conv_in_2(res_1)             

        # body 2
        in_2 = h_2
        h_2 = self.body_2(h_2)   
        h_2 = self.downsample_2(h_2)        

        # residual 2
        res_2 = h_2 + F.avg_pool2d(in_2, 2)           
        if self.auxiliary:
            out_2 = self.conv_out_2(res_2)          
            if x_mid != None:
                x_3 = x_mid
            elif self.auxiliary:
                x_3 = torch.clamp(out_2.detach(), -1.0, 1.0)
            h_3 = self.conv_in_3(x_3)              
        else:
            h_3 = self.conv_in_3(res_2)

        # body 3
        in_3 = h_3
        h_3 = self.body_3(h_3)
        h_3 = self.downsample_3(h_3)                 
        
        # residual 3
        res_3 = h_3 + F.avg_pool2d(in_3, 2)           
        if self.auxiliary:
            out_3 = self.conv_out_3(res_3)
            if x_low != None:
                x_4 = x_low
            elif self.auxiliary:
                x_4 = torch.clamp(out_3.detach(), -1.0, 1.0)
            h_4 = self.conv_in_4(x_4) 
        else:
            out = self.conv_out_3(res_3)                         

        if self.auxiliary:
            # body 4
            in_4 = h_4
            h_4 = self.body_4(h_4)
            h_4 = self.downsample_4(h_4)          

            # residual 4
            res_4 = h_4 + F.avg_pool2d(in_4, 2)                            
            out_4 = self.conv_out_4(res_4)       

        if self.auxiliary:
            return out_1, out_2, out_3, out_4
        else:
            return out
      

class WaveDecoder(nn.Module):
    def __init__(self, hidden_dim, in_channels=128, out_channels=3, num_res_blocks=16):
        super(WaveDecoder, self).__init__() 
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        # 16 x 16 -> 32 x 32
        self.conv_in_1 = nn.Conv2d(in_channels, hidden_dim * 8, 3, 1, 1)
        self.body_1 = ResBlocks(hidden_dim * 8, num_res_blocks)
        self.upsample_1 = Upsampler(hidden_dim * 8)                                                
        self.conv_out_1 = nn.Conv2d(hidden_dim * 8, out_channels, 1, 1)    

        # 32 x 32 -> 64 x 64
        self.conv_in_2 = nn.Conv2d(in_channels, hidden_dim * 4, 3, 1, 1)
        self.body_2 = ResBlocks(hidden_dim * 4, num_res_blocks)
        self.upsample_2 = Upsampler(hidden_dim * 4)                                     
        self.conv_out_2 = nn.Conv2d(hidden_dim * 4, out_channels, 1, 1)

        # 64 x 64 -> 128 x 128
        self.conv_in_3 = nn.Conv2d(in_channels, hidden_dim * 2, 3, 1, 1)
        self.body_3 = ResBlocks(hidden_dim * 2, num_res_blocks)   
        self.upsample_3 = Upsampler(hidden_dim * 2)                                                
        self.conv_out_3 = nn.Conv2d(hidden_dim * 2, out_channels, 1, 1)     

        # 128 x 128 -> 256 x 256
        self.conv_in_4 = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1)
        self.body_4 = ResBlocks(hidden_dim, num_res_blocks)   
        self.upsample_4 = Upsampler(hidden_dim)                                                
        self.conv_out_4 = nn.Conv2d(hidden_dim, out_channels, 1, 1)             

        self.auxiliary = True   
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    def setup_finetune(self):
        self.auxiliary = False
        del self.conv_in_1
        del self.body_1
        del self.upsample_1
        del self.conv_out_1

        del self.conv_out_2
        del self.conv_out_3
        del self.conv_in_3
        del self.conv_in_4
        self.conv_in_3 = nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 2, 1, 1)
        self.conv_in_4 = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 1, 1, 1)

    def forward(self, x, x_low=None, x_mid=None, x_high=None):
        if self.auxiliary:
            # feature extraction
            x_1 = x
            h_1 = self.conv_in_1(x_1)
            # body 1
            in_1 = h_1
            h_1 = self.body_1(h_1)
            h_1 = self.upsample_1(h_1)

            # residual 1
            res_1 = h_1 + F.interpolate(in_1, scale_factor=2, mode='nearest')          
            out_1 = self.conv_out_1(res_1)
            if x_low != None:
                x_2 = x_low
            else:
                x_2 = torch.clamp(out_1.detach(), -1.0, 1.0)
            h_2 = self.conv_in_2(x_2)   
        else:
            h_2 = self.conv_in_2(x)             

        # body 2
        in_2 = h_2
        h_2 = self.body_2(h_2)   
        h_2 = self.upsample_2(h_2)        

        # residual 2
        res_2 = h_2 +  F.interpolate(in_2, scale_factor=2, mode='nearest')            
        if self.auxiliary:
            out_2 = self.conv_out_2(res_2)          
            if x_mid != None:
                x_3 = x_mid
            elif self.auxiliary:
                x_3 = torch.clamp(out_2.detach(), -1.0, 1.0)
            h_3 = self.conv_in_3(x_3)              
        else:
            h_3 = self.conv_in_3(res_2)

        # body 3
        in_3 = h_3
        h_3 = self.body_3(h_3)
        h_3 = self.upsample_3(h_3)                 
        
        # residual 3
        res_3 = h_3 + F.interpolate(in_3, scale_factor=2, mode='nearest')             
        if self.auxiliary:
            out_3 = self.conv_out_3(res_3)
            if x_high != None:
                x_4 = x_high
            elif self.auxiliary:
                x_4 = torch.clamp(out_3.detach(), -1.0, 1.0)
            h_4 = self.conv_in_4(x_4)              
        else:
            h_4 = self.conv_in_4(res_3)

        # body 4
        in_4 = h_4
        h_4 = self.body_4(h_4)
        h_4 = self.upsample_4(h_4)          

        # residual 4
        res_4 = h_4 + F.interpolate(in_4, scale_factor=2, mode='nearest')                             
        out_4 = self.conv_out_4(res_4)       

        if self.auxiliary:
            return out_1, out_2, out_3, out_4
        else:
            return out_4
   