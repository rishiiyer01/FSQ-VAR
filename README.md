# cheapVAR


The goal of this project was to see if I could use a continous VAE instead of  VQVAE for VAR style image autoregression. The idea is that VAR tokens are simply interpolations of the latents, which do not need to be quantized necessarily on input, but *do* need to be quantized on output for cross entropy loss. Theoretically this can easily be done, which means we might need to do no finetuning or very little finetuning to the VAE to get good results. We use Nvidia's COSMOS-TOKENIZER 16x16 for this project.

This project is simple and trained on imagenet.
