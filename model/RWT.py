import numpy as np
import scipy.io as sio  # For saving .mat files
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import torch 
import pywt
device = torch.device( 'cpu')
def fit_regression(x, y, order):
    """
    Fit a regression model.
    Parameters:
        x (ndarray): Predictors. low
        y (ndarray): Dependent variables. high
        order (int): Order of regression.
    
    Returns:
        M (ndarray): Predicted values.
        W (ndarray): Regression coefficients.
    """
    # Build design matrix
    V = torch.ones((x.shape[0], 1)).to(device)
    for j in range(1, order + 1):
        V = torch.hstack((V, x ** j))
    
    # Compute regression coefficients
    W = torch.linalg.pinv(V) @ y#linear model
    M = V @ W  # Predicted values
    return M, W


def rwa1l(im, n):
    """
    Perform one level of the RWA transformation.
    
    Parameters:
        im (ndarray): Input image (2D array).
        n (int): Order of regression.
    
    Returns:
        L (ndarray): Low-pass component.
        H (ndarray): High-pass component.
        w (ndarray): Regression coefficients.
        mse (float): Mean squared error.
    """
    y, z = im.shape
    p = int(np.round(z / 2))
    q = int(np.floor(z / 2))

    # Changed: Added condition
    if p % 2 == 0 and abs(p - (z/2)) > 0.4 and p == q:
        p += 1
            
    #print(p, q)
    H = torch.zeros((y, q), dtype=im.dtype).to(device)
    L = torch.zeros((y, p), dtype=im.dtype).to(device)
    
    for j in range(q):
        H[:, j] = im[:, 2 * j] - im[:, 2 * j + 1]
        L[:, j] = im[:, 2 * j + 1] + torch.floor(H[:, j] / 2) # Changed: Specified torch.floor
    
    if z % 2 != 0:  # Handle odd number of columns
        L[:, p - 1] = im[:, -1]
    
    # Regression
    M, w = fit_regression(L, H, n)
    mse = torch.mean((M - H) ** 2)
    H = H - torch.round(M)#save residuals torch.zeros_like()
    
    return L, H, w, mse


def rwa(im, l, n=1):
    """
    Perform the RWA transformation.
    
    Parameters:
        im (ndarray): Input image (2D array).
        l (int): Number of levels.
        n (int): Order of regression (default = 1).
    
    Returns:
        pim (ndarray): Transformed image.
        WW (list): Regression coefficients for each level.
    """
    y, z = im.shape
    L, H = [], []
    data = im.clone()
    fijo = []
    WW = []  # Side information
    
    for i in range(l):
        L, H, w, mse = rwa1l(data, n)
        WW.append(w)  # Store regression coefficients
        fijo = torch.hstack((H, fijo)) if len(fijo) else H
        #print(fijo.shape,H.shape,L.shape)
        data = L.clone() # Changed: added .clone() to avoid bad memory access
    
    pim = torch.hstack((L, fijo))
    #print(pim.shape,L.shape,fijo.shape)
    return pim, WW

def read_tif(file_path):
    data = tiff.imread(file_path)  # Assuming the tif is already preprocessed
    data = torch.moveaxis(data, 0, -1)  # Ensure shape is (Height, Width, Bands)
    return data

def rwa_transform(raw_image,  dtype, output):
    """
    Perform Regression Wavelet Analysis (RWA) Transform.
    
    Parameters:
        raw_image (str): Path to the input raw image file.
        z (int): Spectral channels.
        y (int): Lines (height).
        x (int): Columns (width).
        dtype (str): Data type ('int16', 'uint16', etc.).
        output (str): Path to the output raw file.
    """
    # Read the image
    with open(raw_image, 'rb') as fid:
        G = read_tif(fid)
    x = G.shape[0] 
    y = G.shape[1]
    z = G.shape[2]
    # Reshape into a 2D array
    im = G.reshape((x * y, z))
    
    # Perform RWA Transform
    l = int(torch.ceil(torch.log2(z)))  # Compute levels   #set level to 1-2
    RWAim, W = rwa(im, l, 1)  # Default order = 1
    # Reshape the transformed image back to spatial dimensions (y, x, z)
    RWAim_reshaped = RWAim.reshape((x, y,z))   
    
    # Normalize and convert to uint8 for PNG saving
    RWAim_normalized = ((RWAim_reshaped - RWAim_reshaped.min()) /
                        (RWAim_reshaped.max() - RWAim_reshaped.min()) * 255).astype(torch.uint8)
    
    # Write the transformed image to output file
    with open(output, 'wb') as fid:
        RWAim.astype(torch.int16).tofile(fid)  # Save as int16
    
    # Save side information to .mat file
    png_output = f"{output}_SI.png"
    Image.fromarray(RWAim_normalized).save(png_output)

    
    print(f"\nImage: {raw_image} \nSize: ({z}, {y}, {x})")
    print(f"Transformed: {output}")
    print(f"Side information: {png_output}")


def generate_regression(x, w, order):
    """
    Generate regression model predictions.
    
    Parameters:
        x (ndarray): Predictors (low-pass component).
        w (ndarray): Regression coefficients.
        order (int): Order of regression.
    
    Returns:
        M (ndarray): Predicted values for the high-pass component.
    """
    # Build design matrix
    V = torch.ones((x.shape[0], 1)).to(device)
    for j in range(1, order + 1):
        V = torch.hstack((V, x ** j))
    
    # Compute predicted values
    M = V @ w
    return M

def inv_rwa1l(L, H, w, n):
    """
    Perform one level of inverse RWA transformation.
    
    Parameters:
        L (ndarray): Low-pass component.
        H (ndarray): High-pass component.
        w (ndarray): Regression coefficients.
        n (int): Order of regression.
    
    Returns:
        im (ndarray): Recovered image at the current level.
    """
    # Perform regression to reconstruct the high-pass component
    M = generate_regression(L, w, n)
    #print(M.shape)
    H =( H + torch.round(M)).to(device)
    
    # Initialize the reconstructed image
    q = H.shape[1]
    p = L.shape[1]
    z = p + q
    im = torch.zeros((L.shape[0], z), dtype=L.dtype).to(device)
    
    # Reconstruct spatial data
    # Changed: operations inside for loop were reversed and had some mistakes
    for j in range(q):
        im[:, 2 * j + 1] = L[:, j] - torch.floor(H[:, j] / 2)
        im[:, 2 * j] = im[:, 2 * j + 1] + H[:, j]
    
    # Handle odd columns
    if z % 2 != 0:
        im[:, 2 * q] = L[:, -1] # Changed: we want L[:, -1] not L[:, p-1]
    
    return im


def inv_rwa(im, l, WW, n):
    """
    Perform the inverse RWA transformation.
    
    Parameters:
        im (ndarray): Transformed image (2D array).
        l (int): Number of levels.
        WW (list): Regression coefficients for each level.
        n (int): Order of regression.
    
    Returns:
        im (ndarray): Recovered image.
    """
    y, z = im.shape
    data = im.clone().to(device)
    
    # Compute dimensions for each level
    P, Q = [], []
    for i in range(l):
        p = int(np.ceil(z / 2))
        q = int(np.floor(z / 2))
        P.append(p)
        Q.append(q)
        z = p
    
    # Reconstruct the image from the levels
    for i in range(l - 1, -1, -1): # TODO: reversed(range(0, l)): 
        p, q = P[i], Q[i]
        L = data[:, :p].to(device)
        H = data[:, p:p + q].to(device)
        
        # Get regression coefficients for the current level
        w = WW[i].to(device)
        
        # Inverse the current level
        aux = inv_rwa1l(L, H, w, n)
        data[:, :p + q] = aux
    
    return data


def apply_wavelet_transform(data, wavelet_name='db4', level=1, num_bands_to_keep=61):
    """
    Apply wavelet transform on high-dimensional spectral data and retain specific bands.
    
    Args:
        data (torch.Tensor): Input tensor of shape (n_bands, height, width).
        wavelet_name (str): Name of the wavelet to use.
        level (int): Decomposition level for wavelet transform.
        num_bands_to_keep (int): Number of wavelet coefficients to retain.
        
    Returns:
        torch.Tensor: Transformed data of shape (n_bands, height, width).
    """
    n_bands, height, width = data.shape
    transformed_data = torch.zeros_like(data)  # Initialize transformed data
    
    for i in range(height):
        for j in range(width):
            # Extract the spectral profile of the pixel
            spectral_profile = data[:, i, j].numpy()
            
            # Perform Discrete Wavelet Transform (DWT)
            coeffs = pywt.wavedec(spectral_profile, wavelet_name, level=level, mode='per')
            
            # Concatenate all coefficients and retain only the first `num_bands_to_keep` coefficients
            all_coeffs = torch.tensor(np.concatenate(coeffs, axis=0))
            retained_coeffs = all_coeffs[:num_bands_to_keep]
            
            # Create zero-padded coefficients
            padded_coeffs = torch.zeros_like(all_coeffs)
            padded_coeffs[:num_bands_to_keep] = retained_coeffs
            
            # Split back into wavelet coefficient bands
            coeff_shapes = [len(c) for c in coeffs]
            split_coeffs = [padded_coeffs[sum(coeff_shapes[:k]):sum(coeff_shapes[:k + 1])].numpy() 
                            for k in range(len(coeff_shapes))]
            
            # Store the padded coefficients back into transformed_data
            transformed_data[:, i, j] = torch.tensor(np.concatenate(split_coeffs, axis=0))
    
    return transformed_data


def apply_inverse_wavelet_transform(transformed_data, wavelet_name='db4', level=1, original_bands=242):
    """
    Apply inverse wavelet transform to reconstruct the original spectral profiles.
    
    Args:
        transformed_data (torch.Tensor): Transformed data of shape (n_bands, height, width).
        wavelet_name (str): Name of the wavelet to use.
        level (int): Decomposition level for wavelet transform.
        original_bands (int): Number of original spectral bands.
        
    Returns:
        torch.Tensor: Reconstructed data of shape (original_bands, height, width).
    """
    n_bands, height, width = transformed_data.shape
    reconstructed_data = torch.zeros((original_bands, height, width))  # Initialize reconstructed data
    
    for i in range(height):
        for j in range(width):
            # Extract the spectral profile of the pixel
            spectral_profile = transformed_data[:, i, j].numpy()
            
            # Perform Inverse Discrete Wavelet Transform (IDWT)
            coeffs = pywt.wavedec(np.zeros(original_bands), wavelet_name, level=level, mode='per')
            coeff_shapes = [len(c) for c in coeffs]
            
            # Split the transformed spectral profile into wavelet coefficient bands
            split_coeffs = [spectral_profile[sum(coeff_shapes[:k]):sum(coeff_shapes[:k + 1])] 
                            for k in range(len(coeff_shapes))]
            
            # Reconstruct the spectral profile
            reconstructed_spectral_profile = pywt.waverec(split_coeffs, wavelet_name, mode='per')[:original_bands]
            
            # Store the reconstructed spectral profile back into reconstructed_data
            reconstructed_data[:, i, j] = torch.tensor(reconstructed_spectral_profile)
    
    return reconstructed_data

