# FSQ-VAR


The goal of this project was to see if I could use a finite scalar quantized VAE instead of  VQVAE for VAR style image autoregression. The idea is that VAR tokens are simply interpolations of the latents, which can easily be quantized by the same finite scalar quantizer as full sized latents. Theoretically this means with little to no finetuning to the FSQ-VAE, we can get a powerful image generation model. This would be great, since the best current discrete image tokenizer is an FSQ-VAE (cosmos).

This project is simple and trained on imagenet for conditional image generation.
