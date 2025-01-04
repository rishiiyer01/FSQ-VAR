import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('cosmos')
from cosmos_tokenizer.image_lib import ImageTokenizer
from model import cosmos_vae

#phase 1: just need to encode two images, var_tokenize, detokenize, match latents
#probably will only need a single epoch

#this should match the original paper exactly: 
class VARTokenizer(nn.Module):
    def __init__(self, resolutions=[(8,8), (16,16), (32,32)]):
        super().__init__()
        self.resolutions = resolutions
        
        
        self.residual_convs = nn.ModuleList([
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
            for _ in range(len(resolutions))
        ])
    
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
            r = torch.round(f_curr).clamp(-0.5, 0.5)  # Assuming FSQ range
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