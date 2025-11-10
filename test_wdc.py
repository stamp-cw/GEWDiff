import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import tifffile
import torch.distributed as dist
from diffusers import UNet2DModel
import datetime
import torch.distributed as dist
dist.init_process_group(backend='nccl')
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from model.edm import ElucidatedDiffusion, UNet2DModelWithBN, UNet2DWithSpectralFidelity,UNet3DWithSpectralFidelity, EMA
from utils.eval import quality_assessment
from model.RWT import rwa, inv_rwa
import argparse
from scipy.ndimage import zoom
import torch
print(torch.__version__)
import numpy as np
import torch
import os
from torchvision import transforms
import tifffile
import torch.distributed as dist
from scipy.ndimage import zoom
from torchvision import transforms
import numpy as np
from model.RWT import rwa, inv_rwa
from sklearn.decomposition import PCA
from skimage import exposure
preprocess = transforms.Compose(
    [
        #transforms.Resize((config.out_size, config.out_size)),
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        #transforms.Normalize([0.5], [0.5]),
    ]
)
preprocess1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Change according to your output size
])
preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),  # Change according to your output size
])
import torch

from scipy.ndimage import zoom

def resize_image_to_quarter(image_np):
    """
    Resize a NumPy array image to one-fourth of its original size using NumPy.

    Parameters:
    - image_np (numpy.ndarray): The input image as a NumPy array (H, W) or (H, W, C).

    Returns:
    - resized_image (numpy.ndarray): The resized image as a NumPy array.
    """
    if image_np.ndim == 2:  # Grayscale image (H, W)
        zoom_factors = (0.25, 0.25)
    elif image_np.ndim == 3:  # RGB or multi-channel image (H, W, C)
        zoom_factors = (1,0.25, 0.25)  # Only resize spatial dimensions
    else:
        raise ValueError("Input image must be 2D (H, W) or 3D (H, W, C).")
    
    resized_image = zoom(image_np, zoom_factors, order=3)  # Use cubic interpolation
    return resized_image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,config,is_train=False):
        #self.data_dir_lq = os.path.join(data_dir+ '/lq')
        self.data_dir_gt = os.path.join(data_dir+ '/gt')
        self.data_dir_mask = os.path.join(data_dir+ '/mask')
        self.data_dir_edge = os.path.join(data_dir+ '/edge')
        #self.data_dir_gt_wt = os.path.join(data_dir+ '/gt_wt')
        #self.data_dir_gt_w = os.path.join(data_dir+ '/gt_w')
        #self.data_dir_lq_wt = os.path.join(data_dir+ '/lq_wt')
        #self.data_dir_lq_w = os.path.join(data_dir+ '/lq_w')
        self.config=config
        self.files = [file for file in os.listdir(self.data_dir_gt) if file.endswith('.tif')]
        self.pca_lr = PCA(n_components=config.compack_bands)
        self.is_train = is_train
        self.l = 0
        # PCA is calculated only during the test phase
        if self.is_train:
            self.fit_pca()

    def fit_pca(self):
        """ Compute high-resolution and low-resolution PCA separately on all training data """
        #all_patches_hr = []
        all_patches_lr = []
        
        for file in self.files:
            img_gt_o = tifffile.imread(os.path.join(self.data_dir_gt, file))*10000
            img = resize_image_to_quarter(img_gt_o)   
            img_gt = img_gt_o / 10000  

            x = self.config.out_size  # 256
            x1 = int(self.config.out_size / 4)  # 64
            z = self.config.bands
            img_lr=preprocess1(img.transpose(1, 2, 0)).reshape(z, x * x).transpose(0, 1)
            if self.config.compack_bands -1>= int(self.config.bands/2):
                self.l = 1
            elif self.config.compack_bands -1>= int(self.config.bands/4):
                self.l = 2
            elif self.config.compack_bands -1>= int(self.config.bands/8):
                self.l = 3
            elif self.config.compack_bands -1>= int(self.config.bands/16):
                self.l = 4
            elif self.config.compack_bands -1>= int(self.config.bands/32):
                self.l = 5
            elif self.config.compack_bands -1>= int(self.config.bands/64):
                self.l = 6
            elif self.config.compack_bands -1>= int(self.config.bands/128):
                self.l = 7
            elif self.config.compack_bands -1>= int(self.config.bands/256):
                self.l = 8
                    
            Rwim, w = rwa(img_lr,int(self.l),1)
            img_lr_hf = Rwim[:,0:self.config.compack_bands]
            img_lr_hf = np.array(img_lr_hf)
            
            all_patches_lr.append(img_lr_hf)

        # Combine all samples and train PCA
        all_patches_lr = np.vstack(all_patches_lr)
        self.pca_lr.fit(all_patches_lr)

        print("PCA calculation completed")
        print("Low-resolution PCA explained variance ratio:", self.pca_lr.explained_variance_ratio_)

    def __getitem__(self, idx):
        file = self.files[idx]
        #img_o=tifffile.imread(os.path.join(self.data_dir_lq, file))
        #img=img_o/ 3000
        img_gt_o= tifffile.imread(os.path.join(self.data_dir_gt, file))*10000
        img_gt=img_gt_o/ 10000
        img_o=resize_image_to_quarter(img_gt_o)
        img=img_o/ 10000
        if self.config.mask == False:
            mask = torch.ones([self.config.out_size, self.config.out_size])
        else:
            mask_file = os.path.splitext(file)[0] + ".npy"
            mask_path = os.path.join(self.data_dir_mask, mask_file)
            mask=np.load(mask_path)
            mask = torch.tensor(mask, dtype=torch.float32)
        if self.config.edge == False:
            edge = torch.ones([self.config.out_size, self.config.out_size])
        else:
            edge_file = os.path.splitext(file)[0] + ".npy"
            edge_path = os.path.join(self.data_dir_edge, edge_file)
            edge=np.load(edge_path)*1.0
        #smooth_edges = cv2.GaussianBlur(edge, (5, 5), sigmaX=1.0)
            edge = torch.tensor(edge, dtype=torch.float32)
        x=self.config.out_size
        z=self.config.bands
        x1=int(self.config.out_size/4)
        im= torch.Tensor(preprocess1(img_gt_o.transpose(1, 2, 0)).reshape(z,x*x).transpose(0,1))
        im_lr= torch.Tensor(preprocess1(img_o.transpose(1, 2, 0)).reshape(z,x*x).transpose(0,1))
        if self.config.compack_bands -1>= int(self.config.bands/2):
            self.l = 1
        elif self.config.compack_bands -1>= int(self.config.bands/4):
            self.l = 2
        elif self.config.compack_bands -1>= int(self.config.bands/8):
            self.l = 3
        elif self.config.compack_bands -1>= int(self.config.bands/16):
            self.l = 4
        elif self.config.compack_bands -1>= int(self.config.bands/32):
            self.l = 5
        elif self.config.compack_bands -1>= int(self.config.bands/64):
            self.l = 6
        elif self.config.compack_bands -1>= int(self.config.bands/128):
            self.l = 7
        elif self.config.compack_bands -1>= int(self.config.bands/256):
            self.l = 8
            
        RWAim, w = rwa(im,self.l,1)
        RWAim_lr, w_lr = rwa(im_lr,self.l,1)
        img_lr_hf = RWAim_lr[:,0:self.config.compack_bands].reshape(x,x,self.config.compack_bands)
        img_lr_hf = np.array(img_lr_hf)
        img_lr_lf = RWAim_lr[:,self.config.compack_bands:z].reshape(x,x,z-self.config.compack_bands)
        img_lr_lf = np.array(img_lr_lf)
        
        img_hr_hf = RWAim[:,0:self.config.compack_bands].reshape(x,x,self.config.compack_bands)
        img_hr_lf = RWAim[:,self.config.compack_bands:z].reshape(x,x,z-self.config.compack_bands)
        img = preprocess(img.transpose(1,2,0))#.permute(1, 0,  2)
        img_gt = preprocess(img_gt.transpose(1,2,0))#.permute(1, 0,  2)
        lr_pca = PCA(n_components=self.config.compack_bands)
        lr_pca.fit(img_lr_hf.reshape(x*x, self.config.compack_bands))
        if self.is_train == False:
            self.pca_lr.fit(img_lr_hf.reshape(x*x, self.config.compack_bands))
        img_lr_pca = lr_pca.transform(img_lr_hf.reshape(x*x, self.config.compack_bands))
        #img_lr_pca = torch.tensor(img_lr_pca, dtype=torch.float32)
        img_lr_input = img_lr_pca[:,0:self.config.pca_bands].reshape(x,x,self.config.pca_bands)
        img_lr_input = preprocess(img_lr_input)
        img_lr_recov = img_lr_pca[:,self.config.pca_bands:self.config.compack_bands].reshape(x,x,self.config.compack_bands-self.config.pca_bands)
        img_hr_pca = lr_pca.transform(img_hr_hf.reshape(x*x, self.config.compack_bands))
        #img_hr_pca = torch.tensor(img_hr_pca, dtype=torch.float32)
        img_hr_input = img_hr_pca[:,0:self.config.pca_bands].reshape(x,x,self.config.pca_bands)
        img_hr_input = preprocess(img_hr_input)
        #prevent nan values
        if torch.isnan(img_lr_input).any():
            print('nan values in img_lr_input')
            img_lr_input[torch.isnan(img_lr_input)] = 0
        if torch.isnan(img_hr_input).any():
            print('nan values in img_hr_input')
            img_hr_input[torch.isnan(img_hr_input)] = 0


        data= {'img_lr': img, 'img_hr': img_gt,'mask': mask,'edge':edge,
              'img_hr_hf': img_hr_input/14000, 'w':w_lr,'img_lr_hf':img_lr_input/14000,'img_lr_recov':img_lr_recov}#[[38, 23, 5],:,:]
        return data
    def __len__(self):
        return len(self.files)
def shift_mean(pred_pca, img_lr_pca):
    """Adjusts mean of each channel in pred_pca to match img_lr_pca."""
    c = pred_pca.shape[2]
    device = pred_pca.device
    for i in range(c):  # Loop through all channels
        shift = img_lr_pca[:, :, i].mean() - pred_pca[:, :, i].mean()
        pred_pca[:, :, i] += shift.to(device)
    return pred_pca
def recover_range(image,mean,std):
    """Recover the image range by the specified mean and standard deviation."""
    h,w,c = image.shape
    image= image.reshape(h*w,c)
    c = image.shape[1]
    for i in range(c):
        image[:,i] = image[:,i]*std[i]+mean[i]
    return image.reshape(h,w,c)
def normalize(image):
    c = image.shape[1]
    for i in range(c):
        image[:,i] = (image[:,i]-image[:,i].mean())/image[:,i].std()/4
    return image
def resize_image_to_quarter(image_np):
    """
    Resize a NumPy array image to one-fourth of its original size using NumPy.

    Parameters:
    - image_np (numpy.ndarray): The input image as a NumPy array (H, W) or (H, W, C).

    Returns:
    - resized_image (numpy.ndarray): The resized image as a NumPy array.
    """
    if image_np.ndim == 2:  # Grayscale image (H, W)
        zoom_factors = (0.25, 0.25)
    elif image_np.ndim == 3:  # RGB or multi-channel image (H, W, C)
        zoom_factors = (1,0.25, 0.25)  # Only resize spatial dimensions
    else:
        raise ValueError("Input image must be 2D (H, W) or 3D (H, W, C).")
    
    resized_image = zoom(image_np, zoom_factors, order=3)  # Use cubic interpolation
    return resized_image
class TrainingConfig:
    def __init__(self, compack_bands=31, pca_bands=3,train_batch_size=2, num_timesteps=500, num_epochs=40, mask=True, edge=True, l1_lambda=0.9, l2_lambda=0.1, l3_lambda=0.01,
                 sigma_min=0.0005, sigma_max=80, sigma_data=0.5, rho=3):
        self.compack_bands = compack_bands
        self.pca_bands = pca_bands
        self.image_size = 64 
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 1  # how many images to sample during evaluation. testing batch size can be only 1, no larger than 1!
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.lr_warmup_steps = 500
        self.save_image_epochs = 10000
        self.save_model_epochs = 30
        self.mixed_precision = 'no'  # 'fp16' for automatic mixed precision
        self.output_dir = '/workspace/diff_sr/result/'  # output directory
        self.out_size = 256 # the generated image resolution
        self.bands = 191
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.num_timesteps = num_timesteps
        self.seed = 0
        self.mask = mask
        self.edge = edge
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l3_lambda = l3_lambda
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

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
    parser.add_argument("--pca_bands",  type=int, default=3, help="Number of PCA bands.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of timesteps.")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--mask", type=str2bool, nargs='?', const=True, default=True, help="Whether to use mask.")
    parser.add_argument("--edge", type=str2bool, nargs='?', const=True, default=True, help="Whether to use edge.")
    parser.add_argument("--l1_lambda", type=float, default=0.9, help="L1 regularization lambda.")
    parser.add_argument("--l2_lambda", type=float, default=0.1, help="L2 regularization lambda.")
    parser.add_argument("--l3_lambda", type=float, default=0.01, help="L3 regularization lambda.")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma value.")
    parser.add_argument("--sigma_max", type=float, default=80, help="Maximum sigma value.")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Sigma data value.")
    parser.add_argument("--rho", type=float, default=7, help="Rho value.")
    return parser.parse_args()
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
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho

    )
    
    print("Training Configuration:")
    print(f"Compack Bands: {config.compack_bands}")
    print(f"PCA Bands: {config.pca_bands}")
    print(f"Timesteps: {config.num_timesteps}")
    print(f"Num Epochs: {config.num_epochs}")
    print(f"Mask and edge: {config.mask} {config.edge}")
    print(f"L1, L2, L3 lambda: {config.l1_lambda} {config.l2_lambda} {config.l3_lambda}")
    print(f"Sigma Min, Sigma Max, Sigma Data, Rho: {config.sigma_min} {config.sigma_max} {config.sigma_data} {config.rho}")
    PATH = "/workspace/diff_sr/result/epoch_"+str(config.num_epochs)+".pth"
    #PATH = "/workspace/diff_sr/result/best.pth"
 
    checkpoint = torch.load(PATH,map_location=lambda storage, loc: storage.cuda(0), weights_only=False)
    
    #model = UNet2DModelWithBN(
        #sample_size=config.out_size,
        #in_channels=config.pca_bands  * 2+1,
        #out_channels=config.pca_bands ,
        #layers_per_block=4,
        #block_out_channels=(128, 128, 256, 256, 512, 512),
        #down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        #up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        #).cuda()
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
    model.load_state_dict(checkpoint['unet_state_dict'], strict=False)
    diffusion = ElucidatedDiffusion(model,image_size=config.out_size, channels=config.pca_bands ,num_sample_steps=config.num_timesteps, l1_lambda=config.l1_lambda, l2_lambda=config.l2_lambda, l3_lambda=config.l3_lambda,
        sigma_min=config.sigma_min, sigma_max=config.sigma_max, sigma_data=config.sigma_data, rho=config.rho)
    #diffusion.load_state_dict(checkpoint['gaussian_diff_config'], strict=False)
    diffusion.eval()
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # Move the model to GPU if available
    diffusion = diffusion.cuda()
    diffusion.eval()
    torch.cuda.empty_cache()
    def calculate_error_map(predicted, ground_truth):
        assert predicted.shape == ground_truth.shape, "Predicted and ground truth images must have the same shape."
        error_map = torch.mean((predicted - ground_truth) ** 2, dim=-1)   
        return error_map
    #image_gt1 = tifffile.imread('/workspace/diff_sr/data/test/gt/1216_hr_md.tif') /10000
    image_gt1 = tifffile.imread('/workspace/diff_sr/data/test2/gt/output_input.tif') 
    #image_gt1 = tifffile.imread('/workspace/diff_sr/data/val2/gt/1016_flip2_hr_md.tif') /10000
    image_lr1 = resize_image_to_quarter(image_gt1)
    test_dataset = Dataset('/workspace/diff_sr/data/test2',config, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    #image_lr = preprocess(image_lr1).permute(1, 0,  2).unsqueeze(0)
    for batch in test_loader:
        image_lr=batch['img_lr_hf']
        image_lr_show=image_lr[0,:,:,:]#[40,30,20]
        image_gt=batch['img_hr_hf']
        img_lr_recov=batch['img_lr_recov']
        W=batch['w']
        image_gt_show=image_gt[0,:,:,:]#[40,30,20]
        mask=batch['mask'].to("cuda")
        edge=batch['edge'].to("cuda")
        print(image_lr_show.shape)
        break
    print(image_lr[0,0,0,0],image_lr.shape,image_gt.shape,img_lr_recov.shape)
    image_lr = image_lr.to("cuda")
    if config.mask & config.edge == False:
        image_recon,images = diffusion.sample(image_lr, batch_size=config.eval_batch_size)
    elif config.mask == False & config.edge == True:
        image_recon,images = diffusion.sample(image_lr, batch_size=config.eval_batch_size)
    elif config.mask == True & config.edge == False:
        image_recon,images = diffusion.sample(image_lr,mask=mask, batch_size=config.eval_batch_size)
    else:
        image_recon,images = diffusion.sample(image_lr,mask=mask,batch_size=config.eval_batch_size)
    RWAim = torch.zeros((config.out_size * config.out_size, config.bands))
    img_pca = torch.zeros((config.out_size * config.out_size, config.compack_bands))
    img_lr_pca = image_recon[0].permute(1, 2,  0)
    img_gt_pca = image_gt[0].permute(1, 2,  0)
    img_lr_hf = image_lr[0].permute(1, 2,  0)
    std_ratio = torch.zeros(config.pca_bands).cuda()
    img_lr_pca_shift = shift_mean(img_lr_pca, image_lr[0].permute(1, 2,  0))###
    for i in range(config.pca_bands):
        std_ratio[i] = img_lr_hf[:,:,i].std()/img_lr_pca_shift[:,:,i].std()*1.12
        # 1.12 is a number to correct the std ratio, it's defined by experiments. usually 1.1-1.5. 
        # the data range is printed out: (img_hsi.min(),img_hsi.max(),image_gt1.min(),image_gt1.max(),data_range)
        # the range for prediced img_hsi should be similar to image_gt1, by adjusting this number. this number is different for each dataset/sensor.
    
    img_pca[:,:config.pca_bands ]=img_lr_pca_shift.reshape(256*256,config.pca_bands )*14000*std_ratio
    img_pca[:,config.pca_bands :]=img_lr_recov[0].reshape(256*256,config.compack_bands-config.pca_bands )
    wavelet_sapce = test_dataset.pca_lr.inverse_transform(img_pca.cpu().numpy())
    RWAim[:,0:config.compack_bands] = torch.Tensor(wavelet_sapce)
    #convert dtype
    RWAim = RWAim.to(torch.double)
    w=W
    for i in range(test_dataset.l):
        w[i]=W[i][0]
    print(W[0].shape)
    img_hsi= inv_rwa(RWAim, test_dataset.l,w, 1) #
    img_hsi = img_hsi.reshape((config.out_size, config.out_size, config.bands))/10000
    data_range = image_gt1.max() - image_gt1.min()
    print(img_hsi.min(),img_hsi.max(),image_gt1.min(),image_gt1.max(),data_range)
    x_true, x_pred =image_gt1.transpose(1, 2,  0).reshape(256,256,config.bands), img_hsi.cpu().numpy()
    x_pred = np.clip(x_pred, image_gt1.min(), image_gt1.max())
    #data_range = max(x_true.max(), x_pred.max()) - min(x_true.min(), x_pred.min())
    result = quality_assessment(x_true, x_pred, data_range=data_range, ratio=4, multi_dimension=True)
    print(result)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    np.save('/workspace/diff_sr/result/resultdiff_'+time+'.npy', x_pred)
    image_recon_show=img_hsi[:,:,[40,30,20]]#[40,30,20]
    fig=plt.figure(figsize=(50, 50))
    fig.add_subplot(3, 3, 1)
    image_lr1_show=image_lr1[[40,30,20],:,:].transpose(1, 2,  0)
    plt.imshow(image_lr1_show)
    plt.title('Low Resolution Image')
    image_recon_show = image_recon_show 
    image_gt_show = (image_gt_show + 1) / 2 
    fig.add_subplot(3, 3, 2)
    plt.imshow(image_recon_show.cpu())
    plt.title('Reconstructed Image')
    fig.add_subplot(3, 3, 3)
    image_GT_show=image_gt1[[40,30,20],:,:].transpose(1, 2,  0)
    plt.imshow(image_GT_show)
    plt.title('Ground Truth Image')
    fig.add_subplot(3, 3, 5)
    plt.imshow((img_lr_pca_shift+1).cpu()[:,:,:3]/2*std_ratio[:3].cpu(),vmin=0, vmax=1)
    plt.title('Reconstructed Image (RWT)')
    fig.add_subplot(3, 3, 6)
    image_gt_show=image_gt_show.permute(1, 2,  0).cpu()[:,:,:3]
    plt.imshow(image_gt_show,vmin=0, vmax=1)
    plt.title('Ground Truth Image (RWT)')

    fig.add_subplot(3, 3, 4)
    image_lr_show = ((image_lr_show + 1) / 2).permute(1, 2,  0).cpu()[:,:,:3]
    plt.imshow(image_lr_show,vmin=0, vmax=1)
    plt.title('Bicubic Image (RWT)')
    fig.add_subplot(3, 3, 7)
    error_bi=calculate_error_map(image_lr_show,image_gt_show)
    plt.imshow(error_bi.cpu(), vmin=0, vmax=1)
    plt.title('Error Bicubic Image')
    plt.colorbar()

    fig.add_subplot(3, 3, 8)
    error_pr=calculate_error_map(image_recon_show.cpu(),image_GT_show)
    plt.imshow(error_pr.cpu(), vmin=0, vmax=1)
    plt.title('Error Image')
    plt.colorbar()

    fig.add_subplot(3, 3, 9)
    plt.plot(img_hsi[125,125,:],label='Reconstructed Image',color='r')
    plt.plot(image_gt1[:,125,125],label='Ground Truth Image',color='b')
    plt.legend()
    plt.title('Spectrum Comparison')

    plt.show()
    plt.savefig('/workspace/diff_sr/result/resultdiff_rwa_pca.png', format='png')

    #plot histogram
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 2, 1)
    #plt.hist(image_lr[0][1,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='b', label='Low Resolution Image')
    #plt.hist(image_recon[0][1,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='r', label='Reconstructed Image')
    plt.hist(image_gt[0][1,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='g', label='Ground Truth Image')
    plt.hist(img_lr_pca_shift[:,:,1].cpu().numpy().flatten(), bins=100, alpha=0.7, color='r', label='Reconstructed PCA')
    plt.legend()
    plt.title('Histogram of Low Resolution Image, Reconstructed Image and Ground Truth Image')
    fig.add_subplot(1, 2, 2)
    #plt.hist(image_lr[0][2,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='b', label='Low Resolution Image')
    #plt.hist(image_recon[0][2,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='r', label='Reconstructed Image')
    plt.hist(image_gt[0][2,:,:].cpu().numpy().flatten(), bins=100, alpha=0.7, color='g', label='Ground Truth Image')
    plt.hist(img_lr_pca_shift[:,:,2].cpu().numpy().flatten(), bins=100, alpha=0.7, color='r', label='Reconstructed Image')
    plt.legend()
    plt.title('Histogram of Low Resolution Image, Reconstructed Image and Ground Truth Image')
    plt.show()
    plt.savefig('/workspace/diff_sr/result/hist_2.png', format='png')
    print('done',std_ratio)