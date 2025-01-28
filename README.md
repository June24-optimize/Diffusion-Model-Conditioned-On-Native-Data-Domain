# Diffusion Model Conditioned On Native Data Domain
This is a method for image reconstruction based on a diffusion model which is conditioned on the native data domain. This method uses Denoising Diffusion Probability Model that apply to multicoil MRI and quantitative MRI reconstruction. This code is an example for single coil MRI reconstruction using diffusion model conditioned on k-space domain. 
DDPM forward and reverse processes are defined onthe native data domain rather than the image domain. 
Gradient descent algorithm is integrated into the diffusion steps to augment feature learning and promote efficient denoising.


![Project Screenshot](./framework.png)

The diffusion steps are embedded into a Unet, specifically in each self-attention layer. This figure shows U-Net network structure used for $\epsilon_{\theta}$:
![Project Screenshot](./Unet.png)


## Citation

If you find this repository useful, please cite:

**Bian, Wanyu, et al.** "Diffusion modeling with domain-conditioned prior guidance for accelerated MRI and qMRI reconstruction." *IEEE Transactions on Medical Imaging*, 2024.

