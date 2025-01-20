from PIL import Image
import numpy as np
import sys
import time
import cmath
from math import cos, sin, pi

### Image processing

def imageLoader(param):
    im = Image.open(param)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image_arr = np.array(im, dtype=np.uint8)
    return image_arr

def saveImage(image_matrix, output_path):
    if image_matrix.dtype != np.uint8:
        image_matrix = np.clip(image_matrix, 0, 255).astype(np.uint8)
    new_image = Image.fromarray(image_matrix)
    new_image.save(output_path)
    print(f"Image saved at {output_path}")


def imageLoader1B(filepath):
    """
    Load an image and convert it into a binary matrix (1-bit representation).
    :param filepath: Path to the input image file
    :return: 2D list representing the binary image
    """
    try:
        # Open image and convert to grayscale
        img = Image.open(filepath).convert("L")  # L mode = 8-bit grayscale

        # Convert image to numpy array
        img_array = np.array(img)

        # Threshold to binary (1 for pixel values > 128, 0 otherwise)
        binary_image = (img_array > 128).astype(np.uint8)

        return binary_image

    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def saveImage1B(image_matrix, output_path):
    """
    Save a binary matrix as a grayscale image (0 -> black, 255 -> white).
    :param image_matrix: 2D list representing the binary image
    :param output_path: Path to save the output image
    """
    try:
        height = len(image_matrix)
        width = len(image_matrix[0])

        # Create a new grayscale image
        new_image = Image.new("L", (width, height))
        for y in range(height):
            for x in range(width):
                new_image.putpixel((x, y), 255 if image_matrix[y][x] == 1 else 0)

        # Save the image
        new_image.save(output_path)
        print(f"Image saved at {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def brightnessChangerFlat(image_matrix, brightness_change, sign_negative):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {i: 0 for i in range(256)}
    
    for i in range(256):
        if sign_negative == False:
            # Increase brightness
            if (255 - i <= brightness_change):
                lookup_table[i] = 255 
            else:
                lookup_table[i] = i + brightness_change
        else:
            # Decrease brightness
            if (i <= brightness_change):
                lookup_table[i] = 0 
            else:
                lookup_table[i] = i - brightness_change

    if len(image_matrix.shape) == 2:
        # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i, j] = lookup_table[image_matrix[i, j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i, j, k] = lookup_table[image_matrix[i, j, k]]
    
    return output_matrix

def brightnessChangerGamma(image_matrix, brightness_change):
    if(brightness_change < 0):
        print("Wrong brightness_changer!")
        return image_matrix
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {i: 0 for i in range(256)}
    for i in range(1,256):
        if (255/i < brightness_change):
            lookup_table[i] = 255
        else:
            lookup_table[i] = i*brightness_change  
    if len(image_matrix.shape) == 2:  
        # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                   output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]] 
    return output_matrix

def contrastChanger(image_matrix, contrast_factor):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {int(i): 0 for i in range(256)}
    for i in range(256):
        if(contrast_factor*(i-128)+128 > 255):
            lookup_table[i] = 255
        elif(contrast_factor * (i - 128) + 128 < 0):
            lookup_table[i] = 0
        else:
            lookup_table[i] = contrast_factor*(i-128) + 128
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]]
    return output_matrix

def negative(image_matrix):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    lookup_table = {int(i): 0 for i in range(256)}
    for i in range(256):
        lookup_table[i] = 255 - i
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                output_matrix[i,j] = lookup_table[image_matrix[i,j]]
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    output_matrix[i,j,k] = lookup_table[image_matrix[i,j,k]]
    return output_matrix

def flipHorizontal(image_matrix):
    height, width, channels = image_matrix.shape
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
                output_matrix[i, j] = image_matrix[i, width - j - 1]
    return output_matrix

def flipVertical(image_matrix):
    height, width, channels = image_matrix.shape
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
                output_matrix[i, j] = image_matrix[height-i-1, j]
    return output_matrix

def flipDiagonal(image_matrix):
    height, width, channels = image_matrix.shape
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
                output_matrix[i, j] = image_matrix[height-i-1, width - j - 1]
    return output_matrix

def shrinkImage(image_matrix, shrink_factor):
    if shrink_factor <= 0 or shrink_factor > 1:
        raise ValueError("Shrink factor must be between 0 and 1.")
    
    height, width = image_matrix.shape[:2]
    new_height = int(height * shrink_factor)
    new_width = int(width * shrink_factor)
    
    if len(image_matrix.shape) == 2:
        output_matrix = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                output_matrix[i, j] = image_matrix[int(i / shrink_factor), int(j / shrink_factor)]
    else:
        channels = image_matrix.shape[2]
        output_matrix = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                output_matrix[i, j, :] = image_matrix[int(i / shrink_factor), int(j / shrink_factor), :]
    return output_matrix

def enlargeImage(image_matrix, enlarge_factor):
    if enlarge_factor<1:
        raise ValueError('Enlarge factor must be greater than 1')
    
    height, width = image_matrix.shape[:2]
    new_height = int(height * enlarge_factor)
    new_width = int(width * enlarge_factor)
    
    if len(image_matrix.shape) == 2:
        output_matrix = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                original_i = min(int(i / enlarge_factor), height - 1)
                original_j = min(int(j / enlarge_factor), width - 1)
                output_matrix[i, j, :] = image_matrix[original_i, original_j, :]
    elif len(image_matrix.shape) == 3:
        channels = image_matrix.shape[2]
        output_matrix = np.zeros((new_height, new_width, channels), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                original_i = min(int(i / enlarge_factor), height - 1)
                original_j = min(int(j / enlarge_factor), width - 1)
                output_matrix[i, j, :] = image_matrix[original_i, original_j, :] 
    else:
        raise ValueError("Unsupported image format.")

    return output_matrix

def alphatf(image_matrix, kernel_size, alpha):
    """
    Parameters:
    - image_matrix: Input image as a NumPy array (grayscale or RGB).
    - kernel_size: Size of the kernel (box) (must be odd).
    - alpha: Fraction of lowest and highest values to discard (0 <= alpha < 0.5).
    """
    if ((alpha < 0) or (alpha >= 0.5)):
        raise ValueError("Alpha must be between 0 and 0.5")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")

    height, width, channels = image_matrix.shape
    filtered_image = image_matrix.copy().astype(np.uint8)

    # Number of pixels to remove on each end
    #d = max(1, int(alpha * kernel_size * kernel_size))
    d = int(alpha * kernel_size * kernel_size)
    border = int((kernel_size - 1) / 2) 

    for i in range(border, height - border):
        for j in range(border, width - border):
            for k in range(channels):
                window =  window = image_matrix[i - border:i + border+1, j - border:j + border+1, k].flatten()
                sorted_window = np.sort(window)
                # Check if d is less than the remaining number of pixels after trimming
                if d < len(sorted_window) // 2:
                    if(d>0):
                        trimmed_window = sorted_window[d:-d]
                    else:
                        trimmed_window = sorted_window
                    filtered_image[i, j, k] = np.mean(trimmed_window)
                else:
                    filtered_image[i, j, k] = np.mean(sorted_window)
                
    return filtered_image.astype(np.uint8)

def contra_harmonic_mean_filter(image_matrix, kernel, P):
    if kernel % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
        
    height, width, channels = image_matrix.shape
    output_matrix = image_matrix.copy().astype(np.uint8)
    border = kernel // 2
    for row in range(border, height - border):
        for col in range(border, width - border):
            for chan in range(channels):   
                window = image_matrix[row - border:row + border + 1, col - border:col + border + 1, chan].flatten()
                mask = window != 0
                den = np.sum(window[mask] ** P) 
                num = np.sum(window[mask] ** (P + 1)) 
                if den != 0:
                    output_matrix[row, col, chan] = num / den
                else:
                    output_matrix[row, col, chan] = 0
    return output_matrix

def mse(original, processed):
    mse = np.mean((original - processed) ** 2)
    return mse

def pmse(original, processed):
    max_value = np.max(original)
    pmse = np.mean((original - processed) ** 2) / (max_value ** 2)
    return pmse

def snr(original, processed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def psnr(original, processed):
    max_value = np.max(original)
    msev = mse(original, processed)
    psnr = 10 * np.log10(max_value ** 2 / msev)
    return psnr

def md(original, processed):
    md = np.max(np.abs(original - processed))
    return md

### Task 2

def histogram(image_matrix, channel=0):
    lookup_table = {i: 0 for i in range(256)}
    if len(image_matrix.shape) == 2:  # Grayscale image
        for value in image_matrix.flatten():
            lookup_table[value] += 1
    else:  # Color image
        for value in image_matrix[:, :, channel].flatten():
            lookup_table[value] += 1
    return np.array([lookup_table[i] for i in range(256)])

def draw_histogram(hist_array):
    width = 256  
    height = 100
    output_matrix = np.zeros((height, width), dtype=np.uint8)
    max_value = np.max(hist_array)
    if max_value > 0:  
        normalized_hist = (hist_array * height / max_value).astype(int)
    else:
        normalized_hist = np.zeros_like(hist_array)
    for x in range(width):
        column_height = normalized_hist[x]
        output_matrix[height - column_height:height, x] = 255
    return output_matrix

def hrayleigh(hist, g_min, g_max, alpha, L=256):
    """
    Perform Rayleigh-based histogram equalization using a precomputed histogram.

    Parameters:
        hist: Precomputed histogram (1D array of size L, where L is the number of gray levels).
        g_min: Minimum brightness in the output image.
        g_max: Maximum brightness in the output image.
        alpha: Scaling factor for the Rayleigh distribution.
        L: Number of gray levels in the image (default is 256).

    Returns:
        The transformed image
    """
    N = np.sum(hist)
    cdf = np.cumsum(hist) / N
    output_matrix = np.zeros(L, dtype=float)
    
    # Apply the Rayleigh-based transformation for each gray level f
    for f in range(L):
        if cdf[f] > 0:
            # Avoid log(0) or negative values by clamping 1 - cdf[f] to a minimum value
            value = max(1 - cdf[f], 1e-10)
            output_matrix[f] = g_min + np.sqrt(-2 * alpha**2 * np.log(value))
    
    # Normalize output_matrix to range [g_min, g_max]
    output_matrix = np.clip(output_matrix, g_min, g_max)

    # Convert to 8-bit integer range [0, 255]
    #output_matrix = ((output_matrix - g_min) / (g_max - g_min) * 255).astype(np.uint8)

    return output_matrix

def u_slaplace(image_matrix, mask_number):
    """
    Universal convolution function that works for any given mask.
    
    Parameters:
        image: Input image to be filtered (2D array).
        mask: Convolution mask (kernel).
    
    Returns:
        Filtered image (2D array).
    """
    start_time = time.time()
    mask1 = [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0] ]

    mask2 = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1] ]

    mask3 = [
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1] ]

    if(mask_number == 1):
        mask = mask1
    elif(mask_number == 2):
        mask = mask2
    else:
        mask = mask3

    height = len(image_matrix)
    width = len(image_matrix[0])
    mask_size = len(mask)
    offset = mask_size // 2
    
    image_matrix = np.array(image_matrix, dtype=np.int32)
    output_image = np.zeros_like(image_matrix)
    
    # Iterate over each pixel in the image
    for p in range(offset, height - offset):
        for q in range(offset, width - offset):
            # Apply convolution on the region of interest
            conv_sum = 0
            for i in range(mask_size):
                for j in range(mask_size):
                    # Get the corresponding pixel from the image
                    conv_sum += image_matrix[p + i - offset][q + j - offset] * mask[i][j]
            
            output_image[p][q] = conv_sum

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    end_time = time.time() 
    print(f"Execution time for u_slaplace: {end_time - start_time:.4f} seconds")

    return output_image

def o_slaplace(image_matrix):
    """
    Optimized convolution function specifically for the Laplacian filter.
    
    Parameters:
        image_matrix: Input image to be filtered (2D array).
    
    Returns:
        Filtered image (2D array).
    """
    start_time = time.time()
    height = len(image_matrix)
    width = len(image_matrix[0])

    image_matrix = np.array(image_matrix, dtype=np.int32)
    
    output_image = np.zeros_like(image_matrix)

    # Perform convolution for each pixel (ignoring borders)
    for p in range(1, height - 1):
        for q in range(1, width - 1):
            # Apply the Laplacian filter (0, -1, 0; -1, 4, -1; 0, -1, 0)
            conv_sum = (
                -image_matrix[p-1][q]    # Top
                -image_matrix[p+1][q]    # Bottom
                -image_matrix[p][q-1]    # Left
                -image_matrix[p][q+1]    # Right
                + 5 * image_matrix[p][q] # Center
            )
            output_image[p][q] = conv_sum

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    end_time = time.time()
    print(f"Execution time for u_slaplace: {end_time - start_time:.4f} seconds")

    return output_image

def osobel(image_matrix):
    """
    Apply the Sobel operator for edge detection in a non-linear fashion using NumPy.
    
    Parameters:
        image_matrix: Input image matrix (2D array).
    
    Returns:
        Filtered image matrix (2D array).
    """
    image_matrix = np.array(image_matrix, dtype=np.uint16)
    height = len(image_matrix)
    width = len(image_matrix[0])
    output_image = np.zeros_like(image_matrix)
    for n in range(1, height - 1):
        for m in range(1, width - 1):
            # Extract the 3x3 neighborhood around the pixel x(n, m)
            A0 = image_matrix[n-1, m-1]
            A1 = image_matrix[n-1, m]
            A2 = image_matrix[n-1, m+1]
            A3 = image_matrix[n, m+1]
            A4 = image_matrix[n+1, m+1]
            A5 = image_matrix[n+1, m]
            A6 = image_matrix[n+1, m-1]
            A7 = image_matrix[n, m-1]
            x_nm = image_matrix[n, m]
            
            # Calculate Sobel gradients X and Y
            X = (A2 + 2 * A3 + A4) - (A0 + 2 * A7 + A6)
            Y = (A0 + 2 * A1 + A2) - (A6 + 2 * A5 + A4)
            
            # Compute the magnitude of the gradient g(n, m)
            g_nm = np.sqrt(X**2 + Y**2)
            
            output_image[n, m] = np.clip(g_nm, 0, 255) 

    return output_image.astype(np.uint8)

def mean(image_matrix, channel=0):
    """Calculate the mean brightness."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = np.sum(np.arange(len(hist)) * hist) / N
    return b

def variance(image_matrix, channel=0):
    """Calculate the variance."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    D_squared = np.sum((np.arange(len(hist)) - b)**2 * hist) / N
    return D_squared

def std_deviation(image_matrix, channel=0):
    """Calculate the standard deviation."""
    D_squared = variance(image_matrix, channel)
    sigma = np.sqrt(D_squared)
    return sigma

def variation_coefficient(image_matrix, channel=0):
    """Calculate the variation coefficient."""
    sigma = std_deviation(image_matrix, channel)
    b = mean(image_matrix, channel)
    bz = sigma / b
    return bz

def asymmetry_coefficient(image_matrix, channel=0):
    """Calculate the asymmetry coefficient."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    sigma = std_deviation(image_matrix, channel)
    bs = (1 / N) * np.sum(((np.arange(len(hist)) - b)**3) * hist) / (sigma**3)
    return bs

def flattening_coefficient(image_matrix, channel=0):
    """Calculate the flattening coefficient."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    b = mean(image_matrix, channel)
    sigma = std_deviation(image_matrix, channel)
    bk = ((1 / N) * np.sum(((np.arange(len(hist)) - b)**4) * hist)) / (sigma**4) - 3
    return bk

def entropy(image_matrix, channel=0):
    """Calculate the entropy."""
    hist = histogram(image_matrix, channel)
    N = np.sum(hist)
    probabilities = hist / N
    bg = -np.sum(probabilities * np.log2(probabilities + 1e-10)) 
    return bg


### Task3

def dilation(image, struct_element):
    """
    Perform morphological dilation on a binary 2D image (1-bit images).
    
    :param image: 2D list representing binary image (0 and 1)
    :param struct_element: 2D list representing binary structural element (0 and 1)
    :return: Dilated image as 2D list
    """
    height = len(image)  # Number of rows in the image
    width = len(image[0])  # Number of columns in the image
    se_rows = len(struct_element)  # Structural element rows
    se_cols = len(struct_element[0])  # Structural element columns

    # Calculate padding size for structural element
    pad_x = se_rows // 2
    pad_y = se_cols // 2

    # Step 1: Pad the original image with zeros to handle border pixels
    padded_image = [[0] * (width + 2 * pad_y) for _ in range(height + 2 * pad_x)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_x][j + pad_y] = image[i][j]

    # Step 2: Initialize the output image
    output_image = [[0 for _ in range(width)] for _ in range(height)]

    # Step 3: Perform Dilation
    for i in range(height):  # Iterate over the image rows
        for j in range(width):  # Iterate over the image columns
            match_found = False
            for m in range(se_rows):  # Iterate over the structural element rows
                for n in range(se_cols):  # Iterate over the structural element columns
                    if struct_element[m][n] == 1:  # Check if SE element is 1
                        if padded_image[i + m][j + n] == 1:  # Check corresponding pixel in padded image
                            match_found = True
                            break
                if match_found:  # Stop searching once a match is found
                    break
            output_image[i][j] = 1 if match_found else 0  # Set output pixel

    output_image = np.array(output_image)
    return output_image

def erosion(image, struct_element):
    """
    Perform morphological erosion on a binary image.
    :param image: 2D binary image (0 and 1)
    :param struct_element: 2D binary structural element (0 and 1)
    :return: Eroded image as a 2D list
    """
    # Image dimensions
    height = len(image)
    width = len(image[0])
    
    # Structural element dimensions
    se_rows = len(struct_element)
    se_cols = len(struct_element[0])

    # Padding sizes
    pad_x = se_rows // 2
    pad_y = se_cols // 2

    # Step 1: Pad the image with zeros
    padded_image = [[0] * (width + 2 * pad_y) for _ in range(height + 2 * pad_x)]
    for i in range(height):
        for j in range(width):
            padded_image[i + pad_x][j + pad_y] = image[i][j]

    # Step 2: Initialize the output image
    output_image = [[0 for _ in range(width)] for _ in range(height)]

    # Step 3: Perform Erosion Operation
    for i in range(height):  # Traverse each pixel in the original image
        for j in range(width):
            match = True  # Assume match initially
            for m in range(se_rows):  # Traverse the structural element
                for n in range(se_cols):
                    if struct_element[m][n] == 1:  # Only consider structural element's 1s
                        if padded_image[i + m][j + n] != 1:  # Check if any mismatch occurs
                            match = False
                            break
                if not match:  # Exit early if no match
                    break
            if match:
                output_image[i][j] = 1  # Set output pixel to 1 if full match found

    output_image = np.array(output_image, dtype=np.uint8)
    return output_image


def opening(image, struct_element):
    """
    Perform morphological opening: erosion followed by dilation.
    :param image: 2D binary image
    :param struct_element: 2D structural element
    :return: Image after opening operation
    """
    eroded = erosion(image, struct_element)
    opened = dilation(eroded, struct_element)
    return opened

def closing(image, struct_element):
    """
    Perform morphological closing: dilation followed by erosion.
    :param image: 2D binary image
    :param struct_element: 2D structural element
    :return: Image after closing operation
    """
    dilated = dilation(image, struct_element)
    closed = erosion(dilated, struct_element)
    return closed

def hmt(image, struct_element):
    """
    Perform Hit-or-Miss Transformation (HMT) on a binary image.
    :param image: 2D binary image
    :param struct_element_B1: Foreground structural element (B1)
    :param struct_element_B2: Background structural element (B2, complement)
    :return: HMT-transformed image
    """
    struct_element_B1 = struct_element[0]
    struct_element_B2 = struct_element[1]
    eroded_B1 = erosion(image, struct_element_B1)
    eroded_B2 = erosion(1 - image, struct_element_B2)
    return eroded_B1 & eroded_B2

def successive_n_transform(image, struct_elements):
    """
    Apply the successive morphological transformations N(A, {B1, ..., Bn})
    :param image: 2D binary image
    :param struct_elements: List of structural elements [B1, B2, ..., Bn]
    :param n: Number of iterations
    :return: Transformed image
    """

    height = len(image)
    width = len(image[0])
    for i in range(1):
        for se in struct_elements:
            new_image = image.copy()
            new_image = hmt(new_image, se)
            for i in range(height):
                for j in range(width):
                    if(new_image[i][j] == 1):
                        image[i][j] = 0
            ##image = np.where(image != new_image, new_image, image)  # Update only where changes occur
    return image
    

    
    '''changes_made = True  # Flag to track if any changes were made

    while changes_made:
        changes_made = False  # Reset flag at the beginning of each iteration

        for se in struct_elements:
            new_image = image.copy()
            new_image = erosion(new_image, se[0])  # Apply erosion with the first element of the pair
            updated_image = np.where(image != new_image, new_image, image)  # Update only where changes occur

            # Check if there was any change
            if not np.array_equal(image, updated_image):
                changes_made = True

            image = updated_image  # Update the image for the next iteration

    return image'''
    

def structural_elements(param):
    # Define sample structural elements (as binary numpy arrays)
    B1 = np.array([[0, 0, 0],
               [0, 1, 1],
               [0, 0, 0]])

    B2 = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 1, 0]])

    B3 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])

    B4 = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

    B5 = np.array([[0, 0, 0],
                [0, 1, 1],
                [0, 1, 0]])

    B6 = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0]])

    B7 = np.array([[0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]])

    B8 = np.array([[0, 0, 0],
                [1, 0, 1],
                [0, 0, 0]])

    B9 = np.array([[0, 0, 1],
                [1, 1, 0],
                [1, 0, 0]])

    B10 = np.array([[0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0]])

    # List of structural elements for N-transform
    structural_elements = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10]
    return structural_elements[param]

def structural_elements_XI(param):
    B1 = np.array([[1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]])

    B2 = np.array([[1, 1, 1],
                [0, 0, 0],
                [0, 0, 0]])

    B3 = np.array([[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]])

    B4 = np.array([[0, 0, 0],
                [0, 0, 0],
                [1, 1, 1]])

    BC = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]])
    
    structural_elements = [
                    [B1, BC], 
                    [B2, BC], 
                    [B3, BC], 
                    [B4, BC]]
    return structural_elements[param]

def structural_elements_XII():

    B1 = np.array([[0, 0, 0],
                [0, 1, 0],
                [1, 1, 1]])

    B1c = np.array([[1, 1, 1],
                [0, 0, 0],
                [0, 0, 0]])

    B2 = np.array([[0, 0, 0],
                [1, 1, 0],
                [1, 1, 0]])

    B2c = np.array([[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]])

    B3 = np.array([[1, 0, 0],
                [1, 1, 0],
                [1, 0, 0]])

    B3c = np.array([[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]])

    B4 = np.array([[1, 1, 0],
                [1, 1, 0],
                [0, 0, 0]])

    B4c = np.array([[0, 0, 1],
                [0, 0, 1],
                [1, 1, 1]])

    B5 = np.array([[1, 1, 1],
                [0, 1, 0],
                [0, 0, 0]])

    B5c = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1]])

    B6 = np.array([[0, 1, 1],
                    [0, 1, 1],
                    [0, 0, 0]])

    B6c = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0]])

    B7 = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [0, 0, 1]])

    B7c = np.array([[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]])

    B8 = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]])

    B8c = np.array([[1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])
    
    B9c  = np.array([[1, 1, 0],
                [0, 1, 0],
                [0, 1, 0]])
    
    B9 = np.array([[0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1]])
    structural_elements = [
        [B1, B1c], 
        [B2, B2c], 
        [B3, B3c], 
        [B4, B4c], 
        [B5, B5c], 
        [B6, B6c], 
        [B7, B7c], 
        [B8, B8c],
        [B9, B9c]]
    return structural_elements

def region_growing(image, seed, threshold, criterion):
    """
    Perform region growing on an image starting from a seed point.

    Parameters:
        image (numpy.ndarray): Input image (H x W x C) where C is the number of channels (e.g., 3 for RGB).
        seed (tuple): A tuple (X, Y) representing the seed point.
        threshold (float): Threshold value for similarity criterion (0 for binary images).
        criterion (int): Distance criterion:
                        0 - Euclidean distance
                        1 - Manhattan distance
                        2 - Maximum absolute difference

    Returns:
        numpy.ndarray: Binary image (H x W) with the grown region marked as 255 (white).
    """
    image = image[:, :, 0]
    image = np.array(image, dtype=np.int16)
    seedX, seedY = seed
    # Initialize the result binary image
    result = np.zeros_like(image, dtype=np.uint8)

    # Seed pixel intensity
    seed_value = image[seedY, seedX]  # Cast seed value to int for arithmetic operations
    result[seedY, seedX] = 255  # Mark seed point as part of the region

    print(f"Seed Value at ({seedX}, {seedY}): {seed_value}")
    

    # Directions for 8-connectivity
    dx = [-1, 1, 0, 0, -1, 1, -1, 1]
    dy = [0, 0, -1, 1, -1, -1, 1, 1]

    # Queue for BFS
    pixel_queue = [(seedX, seedY)]
    
    while pixel_queue:
        x, y = pixel_queue.pop(0)

        # Check all 8 neighbors
        for i in range(8):
            nx = x + dx[i]
            ny = y + dy[i]

            # Ensure the neighbor is within bounds and not yet visited
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and result[ny, nx] == 0:
                neighbor_value = image[ny, nx]  # Pixel value in all channels (for color images)

                # Compute the distance based on the chosen criterion
                distance = 0
                if criterion == 0:  # Euclidean distance (for multi-channel)
                    distance = np.linalg.norm(neighbor_value - seed_value)
                elif criterion == 1:  # Manhattan distance (sum of absolute differences)
                    distance = np.sum(np.abs(neighbor_value - seed_value))
                elif criterion == 2:  # Maximum absolute difference
                    distance = np.max(np.abs(neighbor_value - seed_value))

                # Check if the distance is within the threshold
                if distance <= threshold:
                    result[ny, nx] = 255  # Add to the region
                    pixel_queue.append((nx, ny))  # Add to the queue

    return result

def seed_points():
    # Example of multiple seed points
    return [(50, 50), (100, 100), (320,450), (0,0)]  # List of seed points

# TASK 4

# Function 1: Perform direct and inverse FFT with decimation in the spatial domain
def perform_fft(image):
    """
    Perform the Fast Fourier Transform (FFT) manually and return the Fourier spectrum.

    Args:
        image (ndarray): Input 2D image.

    Returns:
        tuple: (Fourier transform, magnitude spectrum, phase spectrum)
    """
    rows = len(image)
    cols = len(image[0])

    # Precompute twiddle factors for rows and columns
    def precompute_twiddles(N):
        return [[complex(cos(-2 * pi * k * n / N), sin(-2 * pi * k * n / N)) for n in range(N)] for k in range(N)]

    row_twiddles = precompute_twiddles(cols)
    col_twiddles = precompute_twiddles(rows)

    # Helper function for 1D DFT using precomputed twiddles
    def dft_1d(signal, twiddles):
        N = len(signal)
        return [sum(signal[n] * twiddles[k][n] for n in range(N)) for k in range(N)]

    # Apply DFT row-wise
    row_transformed = [dft_1d(row, row_twiddles) for row in image]

    # Apply DFT column-wise
    fft_result = [[0 for _ in range(cols)] for _ in range(rows)]
    for col in range(cols):
        column = [row_transformed[row][col] for row in range(rows)]
        transformed_column = dft_1d(column, col_twiddles)
        for row in range(rows):
            fft_result[row][col] = transformed_column[row]

    # Compute magnitude and phase
    magnitude = [[abs(fft_result[row][col]) for col in range(cols)] for row in range(rows)]
    phase = [[cmath.phase(fft_result[row][col]) for col in range(cols)] for row in range(rows)]

    return fft_result, magnitude, phase

def perform_ifft(fft_data):
    """
    Perform the inverse Fast Fourier Transform (IFFT) manually and return the reconstructed image.

    Args:
        fft_data (ndarray): Fourier transform of the image.

    Returns:
        ndarray: Reconstructed image in the spatial domain.
    """
    rows = len(fft_data)
    cols = len(fft_data[0])

    # Precompute twiddle factors for rows and columns
    def precompute_twiddles(N):
        return [[complex(cos(2 * pi * k * n / N), sin(2 * pi * k * n / N)) for n in range(N)] for k in range(N)]

    row_twiddles = precompute_twiddles(cols)
    col_twiddles = precompute_twiddles(rows)

    # Helper function for 1D IFFT using precomputed twiddles
    def idft_1d(signal, twiddles):
        N = len(signal)
        return [sum(signal[n] * twiddles[k][n] for n in range(N)) for k in range(N)]

    # Apply IFFT column-wise (inverse of column-wise DFT)
    col_ifft = [[0 for _ in range(cols)] for _ in range(rows)]
    for col in range(cols):
        column = [fft_data[row][col] for row in range(rows)]
        transformed_column = idft_1d(column, col_twiddles)
        for row in range(rows):
            col_ifft[row][col] = transformed_column[row]

    # Apply IFFT row-wise (inverse of row-wise DFT)
    ifft_result = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        ifft_result[row] = idft_1d(col_ifft[row], row_twiddles)

    # Normalize the result (divide by total number of elements)
    scale_factor = rows * cols
    reconstructed_image = [[ifft_result[row][col].real / scale_factor for col in range(cols)] for row in range(rows)]

    return reconstructed_image

# Function 2: Visualize Fourier Spectrum
def visualize_spectrum(magnitude, phase):
    """
    Print basic statistics for the magnitude and phase spectrum.

    Args:
        magnitude (ndarray): Magnitude of the Fourier spectrum.
        phase (ndarray): Phase of the Fourier spectrum.
    """
    print("Magnitude Spectrum (min, max):", np.min(magnitude), np.max(magnitude))
    print("Phase Spectrum (min, max):", np.min(phase), np.max(phase))

# Function 3: Frequency-domain filters

def create_filter(image_shape, filter_type, band_params=None, edge_direction=None):
    """
    Create a frequency-domain filter.

    Args:
        image_shape (tuple): Shape of the image (rows, cols).
        filter_type (str): Type of filter ("low-pass", "high-pass", "band-pass", "band-cut", "edge-detect").
        band_params (tuple): Frequency band parameters (low, high) as fractions of max frequency.
        edge_direction (str): Direction for edge detection ("vertical", "horizontal").

    Returns:
        ndarray: Filter mask.
    """
    rows, cols = image_shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)

    # Create filter mask
    if filter_type == "low-pass":
        radius = band_params[0] * max(rows, cols)
        mask = distance <= radius

    elif filter_type == "high-pass":
        radius = band_params[0] * max(rows, cols)
        mask = distance > radius

    elif filter_type == "band-pass":
        low_radius = band_params[0] * max(rows, cols)
        high_radius = band_params[1] * max(rows, cols)
        mask = (distance >= low_radius) & (distance <= high_radius)

    elif filter_type == "band-cut":
        low_radius = band_params[0] * max(rows, cols)
        high_radius = band_params[1] * max(rows, cols)
        mask = ~((distance >= low_radius) & (distance <= high_radius))

    elif filter_type == "edge-detect":
        if edge_direction == "vertical":
            mask = np.abs(x - ccol) >= band_params[0] * cols
        elif edge_direction == "horizontal":
            mask = np.abs(y - crow) >= band_params[0] * rows
        else:
            raise ValueError("Invalid edge direction. Choose 'vertical' or 'horizontal'.")

    else:
        raise ValueError("Invalid filter type.")

    return mask.astype(float)

def apply_filter(fft_data, filter_mask):
    """
    Apply a frequency-domain filter to the Fourier transform of an image.

    Args:
        fft_data (ndarray): Fourier transform of the image.
        filter_mask (ndarray): Frequency-domain filter mask.

    Returns:
        ndarray: Filtered Fourier transform.
    """
    return fft_data * filter_mask

# Function 4: Phase-modifying filter
def create_phase_mask(image_shape, k, l):
    """
    Create a phase-modifying filter mask P(n, m).

    Args:
        image_shape (tuple): Shape of the image (N, M).
        k (int): Integer parameter for the mask.
        l (int): Integer parameter for the mask.

    Returns:
        ndarray: Phase-modifying mask.
    """
    N, M = image_shape
    y, x = np.meshgrid(range(N), range(M), indexing="ij")
    phase = -2 * np.pi * (k * x / M + l * y / N) + (k + l) * np.pi
    mask = np.exp(1j * phase)
    return mask

### CMD commands
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <command> <image_path> [<parameters>] <output_path>")
        sys.exit(1)
    command = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[len(sys.argv)-1]

    # Load the image
    try:
        matrix = imageLoader(image_path)
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        sys.exit(1)
    
    # Handle commands
    if command == '--brightnessFlat':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessFlat <image_path> <brightness_value> <output_path>")
            sys.exit(1)
        try:
            brightness_value = int(sys.argv[3])
            sign_negative = brightness_value < 0  # Determine if brightness is increasing or decreasing
            modified_matrix = brightnessChangerFlat(matrix, abs(brightness_value), sign_negative)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Brightness value must be an integer.")
            sys.exit(1)

    elif command == '--brightnessGamma':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessGamma <image_path> <brightness_factor> <output_path>")
            sys.exit(1)
        try:
            brightness_factor = float(sys.argv[3])
            modified_matrix = brightnessChangerGamma(matrix, brightness_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Brightness factor must be a valid float.")
            sys.exit(1)
    
    elif command == '--contrast':
        if len(sys.argv) != 5:
            print("Usage: python script.py --contrast <image_path> <contrast_factor> <output_path>")
            sys.exit(1)
        try:
            contrast_factor = float(sys.argv[3])
            modified_matrix = contrastChanger(matrix, contrast_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Contrast factor must be a valid float.")
            sys.exit(1)
        
    elif command == '--negative':
        if len(sys.argv) != 4:
            print("Usage: python script.py --negative <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix = negative(matrix)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command=='--hflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --hflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipHorizontal(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command=='--vflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --vflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipVertical(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command=='--dflip':
        if len(sys.argv) !=4:
            print("Usage: python script.py --dflip <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=flipDiagonal(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    elif command == '--shrink':
        if len(sys.argv) != 5:
            print("Usage: python script.py --shrink <image_path> <shrink_factor> <output_path>")
            sys.exit(1)
        try:
            shrink_factor = float(sys.argv[3])
            output_path = sys.argv[4]
            modified_matrix = shrinkImage(matrix, shrink_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--enlarge':
        if len(sys.argv) != 5:
            print("Usage: python script.py --enlarge <image_path> <enlarge_factor> <output_path>")
            sys.exit(1)
        try:
            enlarge_factor = float(sys.argv[3])
            output_path = sys.argv[4]
            modified_matrix = enlargeImage(matrix, enlarge_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--alpha':
        if len(sys.argv) != 6:
                print("Usage: python script.py --alpha <image_path> <kernel_size> <alpha> <output_path>")
                sys.exit(1)
        try:
            kernel_size = int(sys.argv[3])
            alpha = float(sys.argv[4])
            if kernel_size % 2 == 0:
                print("Kernel size must be an odd number.")
                sys.exit(1)
            if ((alpha < 0) or (alpha >= 0.5)):
                print("Alpha must be between 0 and 0.5.")
                sys.exit(1)
            modified_matrix = alphatf(matrix, kernel_size, alpha)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Invalid kernel size or alpha value.")
            sys.exit(1)

    elif command=='--cmean':
        if len(sys.argv) != 6:
            print("Usage: python script.py --cmean <image_path> <kernel> <P> <output_path>")
            sys.exit(1)
        try:
            kernel = int(sys.argv[3])
            P = float(sys.argv[4])
        except ValueError: 
            print("Kernel must be an integer and P must be a float.")
            sys.exit(1)
        try:
            modified_matrix=contra_harmonic_mean_filter(matrix,kernel,P)
            saveImage(modified_matrix,output_path)
            sys.exit(0)
        except Exception as e:
            print(f"Error processing the image: {e}")
            sys.exit(1)
        
    elif command =='--mse':
        if len(sys.argv) != 4:
            print("Usage: python script.py --mse <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(mse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--pmse':
        if len(sys.argv) != 4:
            print("Usage: python script.py --pmse <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(pmse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--snr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --snr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(snr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--psnr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --psnr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(psnr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--md':
        if len(sys.argv) != 4:
            print("Usage: python script.py --md <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imageLoader(output_path)
            print(md(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command == '--test':
        if len(sys.argv) != 5:
            print("Usage: python script.py --md <image_path> <image2_path> <image3_path>")
            sys.exit(1)
        try:
            origin_path = sys.argv[3]
            matrix_o = imageLoader(origin_path)
            matrix_f = imageLoader(output_path)
            print(':Original:')
            print(mse(matrix, matrix_o))
            print(pmse(matrix, matrix_o))
            print(snr(matrix, matrix_o))
            print(psnr(matrix, matrix_o))
            print(md(matrix, matrix_o))
            print("-----")
            print("Filtered:")
            print(mse(matrix, matrix_f))
            print(pmse(matrix, matrix_f))
            print(snr(matrix, matrix_f))
            print(psnr(matrix, matrix_f))
            print(md(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)


    ### Task2

    elif command =='--histogram':
        if len(sys.argv) != 5:
            print("Usage: python script.py --histogram <image_path> <channel> <output_path>")
            sys.exit(1)
        try:
            channel = int(sys.argv[3])
            modified_matrix=histogram(matrix, channel)
            output_matrix = draw_histogram(modified_matrix)
            saveImage(output_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--hrayleigh':
        if len(sys.argv) != 7:
                print("Usage: python script.py --hrayleigh <image_path> <gmin> <gmax> <alpha> <output_path>")
                sys.exit(1)
        try:
            g_min = int(sys.argv[3])
            g_max = int(sys.argv[4])
            alpha = float(sys.argv[5])
            if g_min < 0 or g_max < 0:
                print("G values must be bigger than 0")
                sys.exit(1)
            histogram_matrix = histogram(matrix)
            modified_matrix = hrayleigh(histogram_matrix, g_min, g_max, alpha)
            output_image = np.array([modified_matrix[pixel] for pixel in matrix.flatten()]).reshape(matrix.shape)
            saveImage(output_image, output_path)
        except ValueError:
            print("Error: Invalid g values size or alpha value.")
            sys.exit(1)

    elif command =='--u_slaplace':
        if len(sys.argv) != 5:
            print("Usage: python script.py --u_slaplace <image_path> <mask_number> <output_path>")
            sys.exit(1)
        try:
            mask_number = int(sys.argv[3])
            if (mask_number < 1 or mask_number > 3):
                print("Mask number must be between 1 and 3")
                sys.exit(1)
            modified_matrix=u_slaplace(matrix,mask_number)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--o_slaplace':
        if len(sys.argv) != 4:
            print("Usage: python script.py --u_slaplace <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=o_slaplace(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--osobel':
        if len(sys.argv) != 4:
            print("Usage: python script.py --osobel <image_path> <output_path>")
            sys.exit(1)
        try:
            modified_matrix=osobel(matrix)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--mean':
        if len(sys.argv) != 4:
            print("Usage: python script.py --mean <image_path> <output_path>")
            sys.exit(1)
        try:
            output = mean(matrix)
            print(f"The mean is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--variance':
        if len(sys.argv) != 4:
            print("Usage: python script.py --variance <image_path> <output_path>")
            sys.exit(1)
        try:
            output = variance(matrix)
            print(f"The variance is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--std_deviation':
        if len(sys.argv) != 4:
            print("Usage: python script.py --std_deviation <image_path> <output_path>")
            sys.exit(1)
        try:
            output = std_deviation(matrix)
            print(f"The standard deviation is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--variation_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --variation_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = variation_coefficient(matrix)
            print(f"The variation coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--asymmetry_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --asymmetry_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = asymmetry_coefficient(matrix)
            print(f"The asymmetry coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--flattening_coefficient':
        if len(sys.argv) != 4:
            print("Usage: python script.py --flattening_coefficient <image_path> <output_path>")
            sys.exit(1)
        try:
            output = flattening_coefficient(matrix)
            print(f"The flattening coefficient is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--entropy':
        if len(sys.argv) != 4:
            print("Usage: python script.py --entropy <image_path> <output_path>")
            sys.exit(1)
        try:
            output = entropy(matrix)
            print(f"The entropy is: {output}")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command == '--test2':
        if len(sys.argv) != 4:
            print("Usage: python script.py --test2 <image_path> <output_path>")
            sys.exit(1)
        try:
            print("Calculating metrics...")
            mean_result = mean(matrix)
            print(f"The mean is: {mean_result}")
            
            variance_result = variance(matrix)
            print(f"The variance is: {variance_result}")
            
            std_dev_result = std_deviation(matrix)
            print(f"The standard deviation is: {std_dev_result}")
            
            variation_coeff_result = variation_coefficient(matrix)
            print(f"The variation coefficient is: {variation_coeff_result}")
            
            asymmetry_coeff_result = asymmetry_coefficient(matrix)
            print(f"The asymmetry coefficient is: {asymmetry_coeff_result}")
            
            flattening_coeff_result = flattening_coefficient(matrix)
            print(f"The flattening coefficient is: {flattening_coeff_result}")
            
            entropy_result = entropy(matrix)
            print(f"The entropy is: {entropy_result}")
            
            sys.exit(0)
        except ValueError:
            print("Error processing the image.")
            sys.exit(1)
    ## Task 3

    elif command =='--dilation':
        if len(sys.argv) != 5:
            print("Usage: python script.py --dilation <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = dilation(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--erosion':
        if len(sys.argv) != 5:
            print("Usage: python script.py --erosion <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = erosion(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--opening':
        if len(sys.argv) != 5:
            print("Usage: python script.py --opening <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = opening(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--closing':
        if len(sys.argv) != 5:
            print("Usage: python script.py --closing <image_path> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            struct_number = int(sys.argv[3])
            modified_matrix = closing(matrix, structural_elements(struct_number))
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--hmt':
        if len(sys.argv) != 6:
            print("Usage: python script.py --hmt <image_path> <XI, or XII> <struct_element> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            xval = int(sys.argv[3])
            struct_B = int(sys.argv[4])
            if(xval == 11):
                modified_matrix = hmt(matrix, structural_elements_XI(struct_B))
            else:
                modified_matrix = hmt(matrix, structural_elements_XII()[struct_B])
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--successive_n':
        if len(sys.argv) != 4:
            print("Usage: python script.py --successive_n <image_path> <output_path>")
            sys.exit(1)

        try:
            matrix = imageLoader1B(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            modified_matrix = successive_n_transform(matrix, structural_elements_XII())
            saveImage1B(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)
    
    elif command =='--region_growing':
        if len(sys.argv) != 7:
            print("Usage: python script.py --region_growing <image_path> <seed_points> <threshold> <criterion> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoader(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

        try:
            tseed_points = int(sys.argv[3])
            threshold = int(sys.argv[4])
            criterion = int(sys.argv[5])
            modified_matrix = region_growing(matrix, seed_points()[tseed_points], threshold, criterion)
            saveImage(modified_matrix,output_path)
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    #TASK 4

    elif command =='--Task4':
        if len(sys.argv) != 4:
            print("Usage: python script.py --Task4 <image_path> <output_path>")
            sys.exit(1)
        try:
            matrix = imageLoader(image_path)
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
        try:
            print("Initial FFT")
            fft_result, magnitude, phase = perform_fft(matrix)
            print("Saving Magnitude and Phase")
            saveImage(magnitude,"magnitude.bmp")
            saveImage(phase,"phase.bmp")
            print("Creating Filter and Phase")
            Low_filter = create_filter(matrix.shape,filter_type="low-pass", band_params=(0.2))
            print("Filtering fft")
            filtered_fft = apply_filter(fft_result, Low_filter)
            print("IFFT")
            filtered_image = perform_ifft(filtered_fft)
            saveImage(filtered_image, "Filtered_image.bmp")
            print("Creating Phase mask")
            phase_mask = create_phase_mask(matrix.shape, 2, 3)
            print("Creating fft * phase mask")
            modified_fft = fft_result * phase_mask
            print("IFFT")
            modified_image = perform_ifft(modified_fft)
            saveImage(modified_image, "Modified_image.bmp")
            sys.exit(1)
        except ValueError: 
            print("Error processing the image.")
            sys.exit(1)

    elif command =='--help':
        print("List of commands:")
        print("--brightnessFlat     | Usage: python script.py --brightnessFlat <image_path> <brightness_value> <output_path>")
        print('--brightnessGamma    | Usage: python script.py --brightnessGamma <image_path> <brightness_factor> <output_path>')
        print('--contrast           | Usage: python script.py --contrast <image_path> <contrast_factor> <output_path>')
        print('--negative           | Usage: python script.py --negative <image_path> <output_path>')
        print('--hflip              | Usage: python script.py --hflip <image_path> <output_path>')
        print('--vflip              | Usage: python script.py --vflip <image_path> <output_path>')
        print('--dflip              | Usage: python script.py --dflip <image_path> <output_path>')
        print('--shrink             | Usage: python script.py --shrink <image_path> <shrink_factor> <output_path>')
        print('--enlarge            | Usage: python script.py --enlarge <image_path> <enlarge_factor> <output_path>')
        print('--alpha              | Usage: python script.py --alpha <image_path> <kernel_size> <alpha> <output_path>')
        print('--cmean              | Usage: python script.py --cmean <image_path> <kernel> <P> <output_path>')
        print('--test               | Usage: python script.py --test <orignal_image_path> <noise_image_path> <filtered_image_path>')   
        print("--histogram          | Usage: python script.py --histogram <image_path> <channel> <output_path>")
        print("--hrayleigh          | Usage: python script.py --hrayleigh <image_path> <gmin> <gmax> <alpha> <output_path>")        
        print("--u_slaplace         | Usage: python script.py --u_slaplace <image_path> <mask_number> <output_path>")        
        print("--o_slaplace         | Usage: python script.py --o_slaplace <image_path> <output_path>")        
        print("--osobel             | Usage: python script.py --osobel <image_path> <output_path>")        
        print("--test2              | Usage: python script.py --test2 <image_path> <output_path>")
        print("--dilation           | Usage: python script.py --dilation <image_path> <struct_element> <output_path>")
        print("--erosion            | Usage: python script.py --erosion <image_path> <struct_element> <output_path>")            
        print("--opening            | Usage: python script.py --opening <image_path> <struct_element> <output_path>")            
        print("--closing            | Usage: python script.py --closing <image_path> <struct_element> <output_path>")            
        print("--hmt                | Usage: python script.py --hmt <image_path> <XI, or XII> <struct_element> <output_path>")            
        print("--successive_n       | Usage: python script.py --successive_n <image_path> <output_path>")            
        print("--region_growing       | Usage: python script.py --region_growing <image_path> <seed_points> <threshold> <criterion> <output_path>")            
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

