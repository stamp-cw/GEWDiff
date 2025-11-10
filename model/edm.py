# ---------------------------------------------------------
# Portions of this code are adapted from:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/elucidated_diffusion.py
# Author: Phil Wang (lucidrains)
# Repository: lucidrains/denoising-diffusion-pytorch
# License: MIT
# If used in academic work, please consider citing the repository.
# ---------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch.distributed as dist
from tqdm.auto import tqdm
from torch import nn
from collections import namedtuple
from random import random
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, reduce
from diffusers import UNet2DModel,UNet3DConditionModel
from unet3d import UNet3DModel 
import matplotlib.pyplot as plt
from copy import deepcopy

class UNet2DModelWithBN(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Modify the convolutional layer of each DownBlock and UpBlock and add BatchNorm
        for module in self.down_blocks:
            for layer in module.resnets:
                layer.conv2 = nn.Sequential(
                    layer.conv2,  
                    nn.BatchNorm2d(layer.conv2.out_channels) 
                )
        for module in self.up_blocks:
            for layer in module.resnets:
                layer.conv2 = nn.Sequential(
                    layer.conv2,
                    nn.BatchNorm2d(layer.conv2.out_channels)
                     )


class UNet2DModelWithBN(UNet2DModel):
    def __init__(self, *args, norm_type='instance', **kwargs):
        super().__init__(*args, **kwargs)

        # Replace BatchNorm2d in the model with custom normalization layers
        self.norm_type = norm_type
        self._replace_norm_layers()

    def _replace_norm_layers(self):
        # Walk through the layers and replace BatchNorm2d
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if self.norm_type == 'instance':
                    new_norm = nn.InstanceNorm2d(module.num_features, affine=True)
                elif self.norm_type == 'group':
                    new_norm = nn.GroupNorm(num_groups=32, num_channels=module.num_features)
                else:
                    raise ValueError(f"Unsupported norm type: {self.norm_type}")

                # Replace the module
                parent_module, child_name = self._get_parent_module_and_name(name)
                setattr(parent_module, child_name, new_norm)

    def _get_parent_module_and_name(self, full_name):
        """
        Get the parent module and the child module name from the full module path.
        """
        names = full_name.split('.')
        parent_module = self
        for name in names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module, names[-1]
    
class SpectralFidelityEnhancer(nn.Module):
    """Spectral Fidelity Enhancement Module"""
    def __init__(self, in_channels=200):
        super().__init__()
        # Spectral Attention Mechanism
        self.mid_channels = in_channels // 8
        if in_channels < 8:
            self.mid_channels = 1 # Make sure mid_channels is at least 1
        self.spectral_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, in_channels, 1),
        )
        

    def forward(self, x):
        # Channel Attention Enhancement
        att = self.spectral_att(x)
        # Let att be in the range [0, 1]
        att = torch.clamp(att, min=0.0, max=1.0)*2+0.2
        x_att = x * att
        # # Let att be in the range [0, 1.5]
        x_att = torch.clamp(x_att, min=0.0, max=1.5)
        return x + 1 * x_att  # Residual connection
    
  
class UNet2DWithSpectralFidelity(UNet2DModel):
    """Improved UNet2D model with integrated spectral fidelity enhancement"""
    def __init__(self, *args, 
                 norm_type='instance',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_type = norm_type
        self._replace_norm_layers()
        self._replace_layers_with_spectral_enhancer()
                    
    def _get_parent_module_and_name(self, full_name):
        """
        Get the parent module and the child module name from the full module path.
        """
        names = full_name.split('.')
        parent_module = self
        for name in names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module, names[-1]
    def _replace_norm_layers(self):
        # Walk through the layers and replace BatchNorm2d
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if self.norm_type == 'instance':
                    new_norm = nn.InstanceNorm2d(module.num_features, affine=True)
                elif self.norm_type == 'group':
                    new_norm = nn.GroupNorm(num_groups=32, num_channels=module.num_features)
                else:
                    raise ValueError(f"Unsupported norm type: {self.norm_type}")

                # Replace the module
                parent_module, child_name = self._get_parent_module_and_name(name)
                setattr(parent_module, child_name, new_norm)
    def _replace_layers_with_spectral_enhancer(self):
        """
        Insert SpectralFidelityEnhancer after each Conv2d
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.out_channels  # Need to match the number of SpectralFidelityEnhancer input channels
                # insert SpectralFidelityEnhancer
                new_module = nn.Sequential(
                module,
                SpectralFidelityEnhancer(in_channels=in_channels)  
                )
                # Replace the layers in UNet
                parent_module, child_name = self._get_parent_module_and_name(name)
                setattr(parent_module, child_name, new_module)

   
class UNet3DWithSpectralFidelity(UNet3DModel):
    """Improved UNet2D model with integrated spectral fidelity enhancement"""
    def __init__(self, *args, 
                 norm_type='instance',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_type = norm_type
        self._replace_norm_layers()
        self._replace_layers_with_spectral_enhancer()
                    
    def _get_parent_module_and_name(self, full_name):
        """
        Get the parent module and the child module name from the full module path.
        """
        names = full_name.split('.')
        parent_module = self
        for name in names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module, names[-1]
    def _replace_norm_layers(self):
        # Walk through the layers and replace BatchNorm2d
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if self.norm_type == 'instance':
                    new_norm = nn.InstanceNorm2d(module.num_features, affine=True)
                elif self.norm_type == 'group':
                    new_norm = nn.GroupNorm(num_groups=32, num_channels=module.num_features)
                else:
                    raise ValueError(f"Unsupported norm type: {self.norm_type}")

                # Replace the module
                parent_module, child_name = self._get_parent_module_and_name(name)
                setattr(parent_module, child_name, new_norm)
    def _replace_layers_with_spectral_enhancer(self):
        """
        Insert SpectralFidelityEnhancer after each Conv2d
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.out_channels  # Need to match the number of SpectralFidelityEnhancer input channels
            
                new_module = nn.Sequential(
                module,
                SpectralFidelityEnhancer(in_channels=in_channels)
                )
                parent_module, child_name = self._get_parent_module_and_name(name)
                setattr(parent_module, child_name, new_module)

    
class SpectralFeatureExtractorPretrained(nn.Module):
    def __init__(self, in_channels=242):
        super(SpectralFeatureExtractorPretrained, self).__init__()
        # Loading the pre-trained VGG19 model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)  
        #vgg = models.vgg19(pretrained=True).features
        #self.feature_layers = nn.Sequential(*list(vgg[:16]))  # Use the first 16 layers of VGG
        self.feature_layers = nn.Sequential(*list(vgg.features.children())[:16]) 
        # Since the number of channels of the hyperspectral image does not match, we can add a convolutional layer to adjust the number of input channels.
        self.conv_adjust = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1)

    def forward(self, gen_img, real_img):
        gen_img = self.conv_adjust(gen_img.to(torch.float32))  # Convert hyperspectral images to 3 channels (can be adjusted through learning)
        real_img = self.conv_adjust(real_img.to(torch.float32))
        loss = nn.MSELoss()
        fetures_gen = self.feature_layers(gen_img)
        fetures_real = self.feature_layers(real_img)
        return loss(fetures_gen, fetures_real)
    
def gradient_loss(x_generated, x_real):
    """ Compute gradient consistency loss """
    c = x_generated.shape[1]  
    device = x_generated.device 
    dtype = x_generated.dtype  

    # Create a Sobel filter (1, 1, 3, 3)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    sobel_y = sobel_x.transpose(2, 3)  # y direction Sobel

    # Copy to multichannel to make it work with `groups=c`
    sobel_x = sobel_x.repeat(c, 1, 1, 1)  
    sobel_y = sobel_y.repeat(c, 1, 1, 1)  

    grad_x_gen = F.conv2d(x_generated, sobel_x, padding=1, groups=c)
    grad_y_gen = F.conv2d(x_generated, sobel_y, padding=1, groups=c)
    grad_x_real = F.conv2d(x_real, sobel_x, padding=1, groups=c)
    grad_y_real = F.conv2d(x_real, sobel_y, padding=1, groups=c)
    # Calculating L1 loss
    loss = F.l1_loss(grad_x_gen, grad_x_real) + F.l1_loss(grad_y_gen, grad_y_real) 
    loss = loss / 2  # Divide by 2 to make it smoother
    return loss

def edge_aware_noise_schedule(noise_map, edge_map, sigmas, strength=0.1):
    """
    Calculate edge-aware noise scheduling, reducing noise intensity in edge regions.
    noise_map: The current noise map of the image (61x256x256).
    edge_map: The edge detection map, typically in the range of 0-1, representing edge strength (256x256).
    strength: Controls the degree of noise reduction in edge regions.
    """
    batch_size = noise_map.shape[0]
    max= 36
    sigmas = torch.clamp(sigmas, min=1e-3, max=36.0) / max
    sigmas = sigmas.view(-1, 1, 1, 1)
    scaler = torch.clamp(1 - sigmas, min=0.0, max=1.0)
    # blur edge
    edge_map = F.avg_pool2d(edge_map, 3, stride=1, padding=1)
    edge_map = edge_map.reshape((batch_size,1,256,256))
    edge_scale = 1 - (1-scaler*scaler) * edge_map * strength 
    edge_aware_noise = noise_map * edge_scale
    
    return edge_aware_noise
def compute_gradient(x):
    """
    Computes gradient magnitude image.
    """
    x_grad_kernel = torch.Tensor([
                            [   [1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]
                            ] for i in range(x.shape[1])]).to(x.device)
    x_grad_kernel = x_grad_kernel.view((1,x.shape[1],3,3))

    y_grad_kernel = torch.Tensor([
                            [   [1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]
                            ] for i in range(x.shape[1])]).to(x.device)
    y_grad_kernel = y_grad_kernel.view((1,x.shape[1],3,3))

    x_grad = F.conv2d(x.double(), x_grad_kernel.double(), padding='same')
    y_grad = F.conv2d(x.double(), y_grad_kernel.double(), padding='same')

    gradient_sqrd_norm = (x_grad**2 + y_grad **2)
    return gradient_sqrd_norm / torch.max(gradient_sqrd_norm)

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def extract(a, t, x_shape):
    b, *_ = t.shape
    
    # Ensure t is on the same device as a
    t = t.to(a.device)  # Move t to the device of a
    
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def SAM(x, y):
    cosine_sim = F.cosine_similarity(x, y, dim=1)
    #remove nan values
    if torch.isnan(cosine_sim).any():
        print('nan values in cosine_sim')
        cosine_sim[torch.isnan(cosine_sim)] = 1e-6
    if torch.isnan(x).any():
        print('nan values in x')
        x[torch.isnan(x)] = 1e-6
    return torch.acos(cosine_sim.clamp(-1+1e-6, 1-1e-6)).mean()
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=torch.float64)
    return betas
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, a_min=0, a_max=0.999)

def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=torch.float64) ** 2
    elif beta_schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(num_diffusion_timesteps, s=0.1)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas
class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_size,
        channels=3,
        num_sample_steps=100,
        sigma_min=0.0005,
        sigma_max=50,
        sigma_data=0.5,
        rho=3,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=0.1,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        l1_lambda=1.0,
        l2_lambda=1.0,
        l3_lambda=1.0,
   ):
        super().__init__()
        print("ElucidatedDiffusion initialized")
        #assert net.random_or_learned_sinusoidal_cond
        self.self_condition = None

        self.net = net

        # image dimensions
        self.channels = channels
        self.image_size = image_size

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - torch.linspace(0, 1, num_sample_steps + 1) ** 2)
        self.transition_fn = lambda num_steps, t, transition_pt, k: (1 - torch.tanh((t - transition_pt) * k)) / 2
        self.transition_pt = 0.5
        self.lambda_min = 1e-4 
        self.lambda_max = 1e-1
        self.k = 10
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.perceptual_loss = SpectralFeatureExtractorPretrained(in_channels=channels)
        self.gradient_loss = gradient_loss
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l3_lambda = l3_lambda
    @property
    def device(self):
        return next(self.net.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25
    def time_idx_to_lambda(self, t_idx):
        time_fractions = t_idx.float() / (self.transition_pt * self.num_sample_steps)
        lambdas = (1 - time_fractions) * self.lambda_min + time_fractions * self.lambda_max
        return lambdas
    #def g_gradient(self, x, t_idx):
    #    device = x.device
    #    x_grad = compute_gradient(x)
    #    one_min_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t_idx, x_grad.shape).to(device)
    #    lambdas = self.time_idx_to_lambda(t_idx).to(device)

    #    perona_malik = torch.sqrt(1.0 + (x_grad / lambdas[:, None, None, None])).to(device)
    #    time_transition_scaling = self.transition_fn(self.num_sample_steps, t_idx, self.transition_pt, self.k).to(device)
    #    return one_min_alpha_bar_t / ((perona_malik * (1.0 - time_transition_scaling[:, None, None, None])) + time_transition_scaling[:, None, None, None])
    

    def preconditioned_network_forward(self, noised_images, img_lr, sigma, mask=None,self_cond=None, clamp=False,i=None):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float) or (isinstance(sigma, torch.Tensor) and sigma.ndim == 0):
            sigma = torch.full((batch,), float(sigma), device=device)

        if isinstance(sigma, torch.Tensor) and sigma.ndim == 1 and sigma.shape[0] == 1 and batch > 1:
            sigma = sigma.expand(batch)
        sigma = sigma.clamp(min=1e-5)
        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
        if mask is None:
            mask = torch.ones([batch,1,self.image_size, self.image_size]).to(device, torch.float32)
        mask=mask.reshape((batch,1,self.image_size,self.image_size))

        combined_input = torch.cat([
        self.c_in(padded_sigma) * noised_images, 
        img_lr, 
        mask
        ], dim=1).to(device, torch.float32)

        net_out = self.net(
        combined_input,
        self.c_noise(sigma),
        self_cond
        ).sample

        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)
        #out = shift_mean(out, img_lr)
        return out
    def sample_schedule(self, num_sample_steps = None,device="cuda"):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    

    def noise_distribution(self, batch_size,device="cuda"):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=device)).exp()

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2
 
    def forward(self, img_lr, images,mask=None,edge=None):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'

        images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size,device=device)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        noise = torch.randn_like(images)
        if edge is not None:
            noise = edge_aware_noise_schedule(noise, edge,sigmas= sigmas, strength=0.5)
            #adjusted_noise = noise
            #plt.imshow(adjusted_noise[0].cpu().detach().numpy()[:3,:,:].transpose(1,2,0))
            #plt.colorbar()
            #plt.savefig("/workspace/diff_sr/result/adjusted_sigmas.png")
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images,img_lr, sigmas,mask,i=None)
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(noised_images, img_lr,sigmas,mask, self_cond,i=None)
        #pixel_loss = F.mse_loss(denoised, images, reduction = 'none')
        pixel_loss = 0.5*F.mse_loss(denoised, images, reduction = 'none') + 0.5*SAM(denoised, images)
        #sam_loss = SAM(denoised, images)
        perception_loss = self.perceptual_loss(denoised, images)
        geometric_loss = self.gradient_loss(denoised, images)

        lambda1 = self.l1_lambda
        lambda2 = self.l2_lambda
        lambda3 = self.l3_lambda
        losses = lambda1 * pixel_loss + lambda2 * perception_loss + lambda3 * geometric_loss
        #print(pixel_loss.mean(),perception_loss.mean(),geometric_loss.mean())
        #losses = 0.8*loss1 + 0.1*loss2 + 0.1*loss3
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean(),pixel_loss.mean(),perception_loss.mean(),geometric_loss.mean()
    @torch.no_grad()
    def sample(self, img_lr, batch_size = 1, num_sample_steps = None, mask=None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = img_lr.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps, device=device)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images  = sigmas[0] * torch.randn(shape, device = device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: torch.tensor(sigma).log().neg() if isinstance(sigma, float) else sigma.log().neg()


        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, img_lr, sigmas[i].item(),mask,i=i)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        #images = images.clamp(-1., 1.)
        return denoised, images
    @torch.no_grad()
    def sample2(self, img_lr, batch_size=1, num_sample_steps=None, mask=None):
        device = img_lr.device
        num_sample_steps = num_sample_steps or self.num_sample_steps

        sigmas = self.sample_schedule(num_sample_steps, device=device)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images = sigmas[0] * torch.randn(shape, device=device)

        def sigma_to_t(sigma):
            sigma_tensor = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(sigma, device=device)
            return -sigma_tensor.log()

        eps = 1e-5
        intermediates = []

        for i in range(1, len(sigmas) - 1):
            sigma_i = sigmas[i].item()
            sigma_im1 = sigmas[i - 1].item()
            sigma_ip1 = sigmas[i + 1].item()

            t_i = sigma_to_t(sigma_i)
            t_im1 = sigma_to_t(sigma_im1)
            t_ip1 = sigma_to_t(sigma_ip1)

            h = (t_ip1 - t_i).clamp(min=eps, max=2.0)
            h_last = t_i - t_im1
            r = torch.clamp(h_last / h, min=eps, max=10.0)

            sigma_interp = sigma_i + r.item() * h.item()
            sigma_interp = torch.tensor([sigma_interp], device=device)

            denoised_i = self.preconditioned_network_forward(images, img_lr, sigma_i, mask,  i=i)
            denoised_im1 = self.preconditioned_network_forward(images, img_lr, sigma_im1, mask, i=i-1)

            exp_neg_rh = torch.exp(torch.clamp(-r * h, min=-80, max=80))
            u_i = (sigma_i / sigma_im1) * images - (sigma_i * (exp_neg_rh - 1) / r) * denoised_im1

            denoised_ui = self.preconditioned_network_forward(u_i, img_lr, sigma_interp, mask, i=i)

            D_i = (1 - 1 / (2 * r)) * denoised_i + (1 / (2 * r)) * denoised_ui

            images = (sigma_ip1 / sigma_i) * images - sigma_ip1 * (torch.exp(-h) - 1) * D_i
            images = torch.nan_to_num(images, nan=0.0)
        return images, images
