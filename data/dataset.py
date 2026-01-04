import numpy as np
import torch
import os

from PIL import Image
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
        self.data_dir_lq = os.path.join(data_dir+ '/wrap')
        self.data_dir_gt = os.path.join(data_dir+ '/gt')
        self.data_dir_mask = os.path.join(data_dir+ '/mask')
        self.data_dir_edge = os.path.join(data_dir+ '/edge')
        #self.data_dir_gt_wt = os.path.join(data_dir+ '/gt_wt')
        #self.data_dir_gt_w = os.path.join(data_dir+ '/gt_w')
        #self.data_dir_lq_wt = os.path.join(data_dir+ '/lq_wt')
        #self.data_dir_lq_w = os.path.join(data_dir+ '/lq_w')
        self.config=config
        # self.files = [file for file in os.listdir(self.data_dir_gt) if file.endswith('.tif')]
        self.files = [file for file in os.listdir(self.data_dir_gt) if file.endswith('.png')]
        # self.wrap_files = [file for file in os.listdir(self.data_dir_lq) if file.endswith('.png')]

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
            # img_gt_o = tifffile.imread(os.path.join(self.data_dir_gt, file))
            img_gt_o = np.array(Image.open(os.path.join(self.data_dir_gt, file))).transpose(2, 0, 1).astype(np.float32)
            # img = resize_image_to_quarter(img_gt_o)
            # img = tifffile.imread(os.path.join(self.data_dir_lq, file))
            img = np.array(Image.open(os.path.join(self.data_dir_lq, file))).transpose(2, 0, 1).astype(np.float32)
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
        # file = self.files[idx]
        #img_o=tifffile.imread(os.path.join(self.data_dir_lq, file))
        #img=img_o/ 3000
        # img_gt_o= tifffile.imread(os.path.join(self.data_dir_gt, file))
        # img_gt=img_gt_o/ 10000
        # img_o=resize_image_to_quarter(img_gt_o)
        # img=img_o/ 10000

        file = self.files[idx]
        img_gt_o= np.array(Image.open(os.path.join(self.data_dir_gt, file))).transpose(2, 0, 1).astype(np.float32)
        img_gt=img_gt_o/ 10000
        # img_o=resize_image_to_quarter(img_gt_o)
        img_o = np.array(Image.open(os.path.join(self.data_dir_lq, file))).transpose(2, 0, 1).astype(np.float32)
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