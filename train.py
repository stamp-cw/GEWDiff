import numpy as np
import torch
from tqdm import tqdm
import os
from PIL import Image
import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
# PyTorch parts
import matplotlib.pyplot as plt
# Accelerate parts
from accelerate import Accelerator # main interface, distributed launcher
from accelerate.utils import set_seed # reproducability across devices
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # The GPUs visible to PyTorch (visible indices: 0, 1, 2, 3)
import torch.distributed as dist
dist.init_process_group(backend='nccl')
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F
import torch
print(torch.__version__)

from tqdm import tqdm
from data.dataset import Dataset
from model.edm import ElucidatedDiffusion,UNet2DModelWithBN, UNet2DWithSpectralFidelity,UNet3DWithSpectralFidelity
from utils.eval import quality_assessment
import argparse

class TrainingConfig:
    def __init__(self, compack_bands=31, pca_bands=3,train_batch_size=2, num_timesteps=500, num_epochs=40, mask=True, edge=True, l1_lambda=0.9, l2_lambda=0.1, l3_lambda=0.01,recall=0):
        self.compack_bands = compack_bands
        self.pca_bands = pca_bands
        self.image_size = 64 
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 1  # how many images to sample during evaluation
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.lr_warmup_steps = 500
        self.save_image_epochs = 100000
        self.save_model_epochs = 20
        self.mixed_precision = 'no'  # 'fp16' for automatic mixed precision
        self.output_dir = '/workspace/diff_sr/result/'  # output directory
        self.out_size = 256 # the generated image resolution
        self.bands = 242
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.num_timesteps = num_timesteps
        self.seed = 0
        self.mask = mask
        self.edge = edge
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l3_lambda = l3_lambda
        self.recall = recall

# Parse arguments from the command line
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser(description="Train the diffusion model with specified parameters.")
    parser.add_argument("--compack_bands", type=int, default=31, help="Number of compack bands.")
    parser.add_argument("--pca_bands", type=int, default=3, help="Number of PCA bands.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of timesteps.")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--mask", type=str2bool, nargs='?', const=True, default=True, help="Whether to use mask.")
    parser.add_argument("--edge", type=str2bool, nargs='?', const=True, default=True, help="Whether to use edge.")
    parser.add_argument("--l1_lambda", type=float, default=0.9, help="L1 loss weight.")
    parser.add_argument("--l2_lambda", type=float, default=0.1, help="L2 loss weight.")
    parser.add_argument("--l3_lambda", type=float, default=0.01, help="L3 loss weight.")
    parser.add_argument("--recall", type=int, default=0, help="Whether to recall the model.")
    return parser.parse_args()
def train_step(batch, step, diffusion, optimizer, lr_scheduler,scaler,progress_bar,global_step,accelerator):
    with accelerator.accumulate(diffusion):
        device = accelerator.device#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        x = batch['img_lr_hf'].to(device)
    ##clean_images = x
    ##noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = x.shape[0]
        t = torch.randint(0, config.num_timesteps, (bs,), device=x.device).long()
        y = batch['img_hr_hf'].to(device)
        mask = batch['mask'].to(device)
        edge = batch['edge'].to(device)
        x_r=y#diffusion.img2res(y,x)
    ##noisy_images = noise_scheduler.add_noise(clean_images, noise, t)
    ##noise_pred = model(noisy_images, t, return_dict=False)[0]
        if config.mask & config.edge == False:
            loss,loss1,loss2,loss3 = diffusion(x,x_r,None,None)
        elif config.mask == True & config.edge == False:
            loss,loss1,loss2,loss3= diffusion(x,x_r,mask,None)
        elif config.mask == False & config.edge == True:
            loss,loss1,loss2,loss3= diffusion(x,x_r,None,edge)
        else:
            loss,loss1,loss2,loss3= diffusion(x,x_r,mask,edge)
    #loss = diffusion.loss_fn(noise_pred, noise, t)
    ###loss = diffusion.loss_fn(x_recon, y, t)
        if scaler != None:
            scaler.scale(loss)  # Scale the loss for backward pass
            accelerator.backward(loss.to("cuda", torch.float64))
            accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)  # Update the model parameters
            scaler.step(lr_scheduler)
            scaler.update()
        else:
            accelerator.backward(loss.to("cuda", torch.float64))
            accelerator.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        #optimizer.zero_grad()
    #dist.destroy_process_group()
    progress_bar.update(1)
    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,"loss1":loss1.detach().item(),"loss2":loss2.detach().item(),"loss3":loss3.detach().item()}
    progress_bar.set_postfix(**logs)
    accelerator.log(logs, step=global_step)
    global_step += 1
    return loss,global_step, loss1, loss2, loss3

def train_epoch(train_loader, diffusion, optimizer, lr_scheduler,scaler, global_step, accelerator, epoch):
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    #diffusion.train()
    diffusion.train()
    train_losses = []
    train_losses1 = []
    train_losses2 = []
    train_losses3 = []

    for i, batch in enumerate(train_loader):
        loss, global_step,loss1,loss2,loss3 = train_step(batch, i, diffusion, optimizer, lr_scheduler,scaler, progress_bar, global_step, accelerator)
        train_losses.append(loss.item())
        train_losses1.append(loss1.item())
        train_losses2.append(loss2.item())
        train_losses3.append(loss3.item())
    return np.mean(train_losses), global_step, train_losses1, train_losses2, train_losses3


def eval_step(batch, diffusion, config):
    with torch.no_grad():  # Disable gradient tracking
        x = batch['img_lr_hf'].to(diffusion.device)
        y = batch['img_hr_hf'].to(diffusion.device)
        mask = batch['mask'].to(diffusion.device)
        edge = batch['edge'].to(diffusion.device)
        x_r = y

        t = torch.randint(0, config.num_timesteps, (x_r.shape[0],), device=diffusion.device)
        if config.mask & config.edge == False:
            loss,loss1,loss2,loss3 = diffusion(x,x_r,None,None)
        elif config.mask == True & config.edge == False:
            loss,loss1,loss2,loss3= diffusion(x,x_r,mask,None)
        elif config.mask == False & config.edge == True:
            loss,loss1,loss2,loss3= diffusion(x,x_r,None,edge)
        else:
            loss,loss1,loss2,loss3= diffusion(x,x_r,mask,edge)
        
        print(f'loss1: {loss1.item()}',f'loss2: {loss2.item()}',f'loss3: {loss3.item()}')
    return loss

def eval_epoch(val_loader, diffusion, config):
    #diffusion.eval()
    diffusion.eval()
    val_losses = []
    for batch in tqdm(val_loader):
        loss= eval_step(batch, diffusion, config)
        val_losses.append(loss.item())
    return np.mean(val_losses)

    
def save_checkpoint(unet, gaussian_diff, optimizer, epoch, loss):
    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'gaussian_diff_config': gaussian_diff.state_dict(),  # Assuming you have a method to get config
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    filename=os.path.join(config.output_dir, f'epoch_{epoch + 1}.pth')
    if dist.get_rank() == 0:
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at {filename}")
#os.makedirs(config.output_dir, exist_ok=True)
def train_loop(config,mixed_precision="fp16", seed=42):
    set_seed(42)
    best_val_loss = float('inf')
    global_step = 0
    accelerator = Accelerator(mixed_precision=mixed_precision)

    train_dataset = Dataset('/workspace/diff_sr/data/train',config)
    val_dataset = Dataset('/workspace/diff_sr/data/val2',config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=1)
    # 检查是否启用了多 GPU
    if accelerator.state.deepspeed_plugin is not None:
        print(f"Using {accelerator.state.num_processes} GPUs")
    else:
        print(f"Using {accelerator.state.local_process_index + 1} GPU(s)")

# 或者直接查看设备
    print(f"Using device: {accelerator.device}")
    with accelerator.main_process_first():
        
        #model = UNet2DModelWithBN(
        #sample_size=config.out_size,
        #in_channels=config.pca_bands * 2+1,
        #out_channels=config.pca_bands ,
        #layers_per_block=4,
        #block_out_channels=(128, 128, 256, 256, 512, 512),
        #down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        #up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        #).to("cuda", torch.float32)
        #model = UNet2DWithSpectralFidelity(
        #sample_size=config.out_size,
        #in_channels=config.pca_bands * 2+1,
        #out_channels=config.pca_bands ,
        #norm_type='group',
        #layers_per_block=4,
        #block_out_channels=(128, 128, 256, 256, 512, 512),
        #down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        #up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        #).to("cuda", torch.float32)
        model = UNet3DWithSpectralFidelity(
        sample_size=config.out_size,
        in_channels=config.pca_bands * 2+1,
        out_channels=config.pca_bands ,
        norm_type='group',
        layers_per_block=4,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        ).to("cuda", torch.float32)

        #model = nn.DataParallel(model).to(accelerator.device)

    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs")

    diffusion = ElucidatedDiffusion(model,image_size=config.out_size, channels=config.pca_bands ,num_sample_steps=config.num_timesteps, l1_lambda=config.l1_lambda, l2_lambda=config.l2_lambda, l3_lambda=config.l3_lambda)
    #print(diffusion.parameters().dtype)
#model=nn.DataParallel(model).to(device)#,device_ids = gpu_ids,output_device=output_device)
#model.to(device)
#torch.cuda.set_device(torch.cuda.current_device())
#model = DDP(model, device_ids=[rank])
#model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config.learning_rate, weight_decay=0.001)


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1, (epoch + 1) / config.lr_warmup_steps))
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )

    scaler = None# torch.cuda.amp.GradScaler() if mixed_precision == 'fp16' else None
    diffusion, optimizer, train_loader, lr_scheduler = accelerator.prepare(diffusion, optimizer, train_loader, lr_scheduler)
    all_loss1 = []
    all_loss2 = []
    all_loss3 = []
    if args.recall > 0:
        model.load_state_dict(torch.load('/workspace/diff_sr/result/epoch_'+str(args.recall)+'.pth')['unet_state_dict'], strict=False)
        diffusion.load_state_dict(torch.load('/workspace/diff_sr/result/epoch_'+str(args.recall)+'.pth')['gaussian_diff_config'], strict=False)
        optimizer.load_state_dict(torch.load('/workspace/diff_sr/result/epoch_'+str(args.recall)+'.pth')['optimizer_state_dict'])
        epoch = torch.load('/workspace/diff_sr/result/epoch_'+str(args.recall)+'.pth')['epoch']
        best_val_loss = torch.load('/workspace/diff_sr/result/epoch_'+str(args.recall)+'.pth')['loss']
        global_step = epoch * len(train_loader)

        print(f"Recall the model at epoch {epoch}, best val loss: {best_val_loss}")
    for epoch in range(config.recall, config.num_epochs):
        train_loss, global_step,loss1_steps,loss2_steps,loss3_steps = train_epoch(train_loader, diffusion, optimizer, lr_scheduler,scaler, global_step, accelerator, epoch)
        val_loss = eval_epoch(val_loader, diffusion, config)
        all_loss1.extend(loss1_steps)
        all_loss2.extend(loss2_steps)
        all_loss3.extend(loss3_steps)
        print(f'Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
            'unet_state_dict': model.state_dict(),
            #'gaussian_diff_config': diffusion.state_dict(),  # Assuming you have a method to get config
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': best_val_loss,
            }
            filename=os.path.join(config.output_dir, f'best.pth')
            if dist.get_rank() == 0:
                torch.save(checkpoint, filename)
                print(f"Checkpoint saved at {filename}")

        if (epoch + 1) % config.save_image_epochs == 0:
            x = next(iter(val_loader))['img_lr_hf'].to(accelerator.device)
            mask = next(iter(val_loader))['mask'].to(accelerator.device)
            edge = next(iter(val_loader))['edge'].to(accelerator.device)
            y = next(iter(val_loader))['img_hr_hf'].to(accelerator.device)
            if config.mask & config.edge == False:
                x_sample, images = diffusion.sample(x)
            elif config.mask == True & config.edge == False:
                x_sample, images = diffusion.sample(x,mask)
            elif config.mask == False & config.edge == True:
                x_sample, images = diffusion.sample(x)
            else:
                x_sample, images = diffusion.sample(x,mask)
            x_sample = (x_sample.clamp(-1, 1) + 1) / 2
            x_sample = x_sample.permute(0, 2, 3, 1).cpu().detach().numpy()
            x_sample = (x_sample * 255).astype(np.uint8)
            Image.fromarray(x_sample.reshape(-1, x_sample.shape[-1])).save(os.path.join(config.output_dir, f'sample_{epoch + 1}.png'))
            result = quality_assessment(x_sample.cpu().numpy(), y.cpu().numpy(), 1, 4)
            print(f'Quality assessment: {result}')

        if (epoch + 1) % config.save_model_epochs == 0:
            save_checkpoint(model, diffusion, optimizer, epoch, best_val_loss)
        # 绘制损失曲线
        if len(all_loss1) > 0 and len(all_loss2) > 0 and len(all_loss3) > 0:
            plt.figure()
            plt.plot(all_loss1, label='loss1')
            plt.plot(all_loss2, label='loss2')
            plt.plot(all_loss3, label='loss3')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(config.output_dir, 'loss.png'))
            plt.close()
        else:
            print("Warning: Loss lists are empty, skipping plot.")
    accelerator.wait_for_everyone() 
    diffusion = accelerator.unwrap_model(diffusion)
# Example usage
    save_checkpoint(model, diffusion, optimizer, epoch, best_val_loss)
if __name__ == "__main__":
    args = parse_args()
    
    # Create the TrainingConfig object with command-line arguments
    config = TrainingConfig(
        compack_bands=args.compack_bands,
        pca_bands=args.pca_bands,
        train_batch_size=args.train_batch_size,
        num_timesteps=args.timesteps,
        num_epochs=args.num_epochs,
        mask=args.mask,
        edge=args.edge,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        l3_lambda=args.l3_lambda,
        recall=args.recall

    )
    
    print("Training Configuration:")
    print(f"Compack Bands: {config.compack_bands}")
    print(f"PCA Bands: {config.pca_bands}")
    print(f"Timesteps: {config.num_timesteps}")
    print(f"Num Epochs: {config.num_epochs}")
    print(f"Mask and edge: {config.mask} {config.edge}")
    print(f"L1, L2, L3 lambda: {config.l1_lambda} {config.l2_lambda} {config.l3_lambda}")
    print(f"Recall: {config.recall}")
    train_loop(config=config,mixed_precision="fp16", seed=42)
    dist.destroy_process_group()