import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('cosmos')
from cosmos_tokenizer.image_lib import ImageTokenizer
from model import cosmos_vae
from model import FSQConverter

#phase 1: just need to encode two images, var_tokenize, detokenize, match latents
#probably will only need a single epoch

#this should match the original paper exactly with the exception of resolutions and quantization method
#original paper used 10 scales, we use 8
class VARTokenizer(nn.Module):
    def __init__(self, resolutions=[(1,1),(2,2),(4,4),(6,6) (8,8),(10,10),(12,12),(16,16)]):
        super().__init__()
        self.resolutions = resolutions
        
        
        self.residual_convs = nn.ModuleList([
            nn.Conv2d(6,6, kernel_size=3, padding=1).to(torch.bfloat16) #channels=6 because of discrete cosmos
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