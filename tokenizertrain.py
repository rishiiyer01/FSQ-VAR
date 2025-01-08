import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('cosmos')
from cosmos_tokenizer.image_lib import ImageTokenizer
from model import cosmos_vae
from var_tokenizer import VARTokenizer
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

#we can train this on a single epoch of imagenet with a single h100, no fsdp needed

#there are some hardcoded numbers that I will update, be warned

from functools import reduce
import operator
import numpy as np

from torch.distributions import Categorical

from torchvision import transforms

import wandb
from PIL import Image
from datasets import load_dataset
import sys


model_name = "Cosmos-Tokenizer-DI16x16"
encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit').to('cuda')

#all we are asking the model to learn is reconstructing the latents from the interpolated VAR tokens, theoretically mse here should be fine
wandb.init(project="var-tokenizer-imagenet", name="training-run")
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        ds = load_dataset("evanarlian/imagenet_1k_resized_256")
        self.dataset = ds[split]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        
    def __getitem__(self, index):
        image_data = self.dataset[index]['image']
        condition=self.dataset[index]['label']
        image = self.transform(image_data)
        
        return image,condition

    def __len__(self):
        return len(self.dataset)

dataset = ImageDataset()

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=128,  
    shuffle=True,
    num_workers=12,
    pin_memory=True,  
    persistent_workers=True  
)




    


lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).to('cuda')
model_name = "Cosmos-Tokenizer-DI16x16"

decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit').to('cuda')

def loss_fn(latents,target_latents,image):
    total_loss=F.mse_loss(latents,target_latents) #simple mse should be fine here
    with torch.no_grad():
        out=decoder.decode(latents)
        lpips_loss_var=lpips(out,image)
        lpips_loss_original=lpips(target_latents,image)
    
    return total_loss,lpips_loss_var,lpips_loss_original
num_epochs=1
model = VARTokenizer()
#model.load_state_dict(torch.load('spectral_model_latent.pth'))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
max_grad_norm = 2.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).to(torch.bfloat16)

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

print('param count', count_params(model))
from tqdm import tqdm
for epoch in range(num_epochs):
    model.train()
    total = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images,condition = batch #condition is here only because I reuse the dataloader class in phase 2
        images = images.to(device)
        with torch.no_grad():
            (indices,encoded)=encoder.encode(images.to(torch.bfloat16))
            #encoded=encoded.to(torch.float) #this depends on whether you want to train the var tokenizer in bfloat16 or float32, cosmos is native bfloat16
        images=images.to('cuda')
        optimizer.zero_grad()
        output = model.encoder(encoded)
        output=model.decoder(output)
        total_loss,lpips_loss_var,lpips_loss_original = loss_fn(output, encoded,images) #reconstruction goal
        
        if not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += total_loss.item()
            
            wandb.log({
                'total_loss': total_loss.item(),
                'lpips_original': lpips_loss_original.item(),
                'lpips_var': lpips_loss_var.item(),
            })
    
    avg_loss = total / len(dataloader)
    wandb.log({'avg_loss': avg_loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'var_tokenizer.pth')
print("Training completed and model saved.")