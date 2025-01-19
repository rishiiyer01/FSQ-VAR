import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('cosmos')#git cloned repo rename
from cosmos_tokenizer.image_lib import ImageTokenizer
from model import cosmos_vae
from model import FSQConverter

#phase 1: just need to encode two images, var_tokenize, detokenize, match latents
#probably will only need a single epoch

#this should match the original paper exactly with the exception of resolutions and quantization method
#original paper used 10 scales, we use 8
#since we only have a embed_dim of 6, we use a residual block instead of a simple linear conv
class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion_factor=8):
        super().__init__()
        hidden_dim = channels * expansion_factor
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1).to(torch.bfloat16)
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1).to(torch.bfloat16)
        self.act = nn.GELU()  # or nn.SiLU()
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        return h

        
class VARTokenizer(nn.Module):
    def __init__(self, resolutions=[(1,1),(2,2),(4,4),(6,6), (8,8),(10,10),(12,12),(16,16)]):
        super().__init__()
        self.resolutions = resolutions
        
        
        self.residual_convs = nn.ModuleList([
            ResidualBlock(6).to(torch.bfloat16) #channels=6 because of discrete cosmos
            for _ in range(len(resolutions))
        ])
        #self.fsqconverter=FSQConverter() #converts latents to indices or indices to latents by the specified nvidia fsq
        model_name="Cosmos-Tokenizer-DI16x16"
        self.quantizer=ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit').to('cuda')._enc_model.quantizer
        
    def encode(self, latents):
        """
        Args:
            latents: Original VAE latent [B, C, H, W]
        Returns:
            List of quantized tokens at each scale
        """
        f = latents
        tokens = []
        
        
        for i, (h, w) in enumerate(self.resolutions):
            # Interpolate to current scale
            f_curr = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
            
            # Quantize
            
            tup=self.quantizer(f_curr.to(torch.float32)) #fsq works best in float32
            r=tup[1].to(torch.bfloat16)
            tokens.append(r)
            
            
            if i < len(self.resolutions) - 1:
                z = self.residual_convs[i](r)
                z = F.interpolate(z, size=f.shape[-2:], mode='bilinear', align_corners=False)
                f = f - z
                
        return tokens
    
    def decode(self, tokens):
        """
        Args:
            tokens: List of quantized tokens at each scale
        Returns:
            Reconstructed latent
        """
        f = torch.zeros_like(tokens[-1])
        
        for i, r in enumerate(tokens):
            z = self.residual_convs[i](r)
            z = F.interpolate(z, size=f.shape[-2:], mode='bilinear', align_corners=False)
            f = f + z
            
        return f





class varmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaleTokenizer=VARTokenizer()
        model_name="Cosmos-Tokenizer-DI16x16"
        self.encoder=ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit').to('cuda')
        self.decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit').to('cuda')
        self.fsq=FSQConverter()
        
    def forward(self,x):
        #x is an image of shape (B,C,256,256) for imagenet
        indices,encoded=self.encoder.encode(x.to(torch.bfloat16))
        encodedscaled=self.scaleTokenizer.encode(encoded)
        latents=self.scaleTokenizer.decode(encodedscaled)
        
        ind=self.fsq.latents_to_indices(latents)
        image=self.decoder.decode(ind)
        
        target_latents=encoded
        
        return image,latents,target_latents #latents for latent loss, image for reconstruction/perceptual loss






        