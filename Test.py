from pytorch_diffusion import Diffusion
import torchvision

diffusion = Diffusion.from_pretrained("ema_cifar10")
samples = diffusion.denoise(36)
samples = (samples+1)*0.5
torchvision.utils.save_image(samples, 'cifar10_generated_sp_imgs.png', nrow=6)

# diffusion = Diffusion.from_pretrained("ema_lsun_church")
# samples = diffusion.denoise(4, n_steps=1000)
# print(samples.shape)
# torchvision.utils.save_image(samples, 'lsun_church_generated_imgs.png', nrow=2)

