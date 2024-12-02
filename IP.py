from PIL import Image
import numpy as np
import sys
import time

### Image processing

def imagineLoader(param):
    im = Image.open(param)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image_arr = np.array(im, dtype=np.uint8)
    return image_arr

def saveImage(image_matrix, output_path):
    image_matrix = np.clip(image_matrix, 0, 255).astype(np.uint8)
    new_image = Image.fromarray(image_matrix)
    new_image.save(output_path)
    print(f"Image saved at {output_path}")

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
        hist (numpy.ndarray): Precomputed histogram (1D array of size L, where L is the number of gray levels).
        g_min (float): Minimum brightness in the output image.
        g_max (float): Maximum brightness in the output image.
        alpha (float): Scaling factor for the Rayleigh distribution.
        L (int): Number of gray levels in the image (default is 256).

    Returns:
        numpy.ndarray: The transformed image, with the same size as the histogram.
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
        image (list of lists): Input image to be filtered (2D array).
        mask (list of lists): Convolution mask (kernel).
    
    Returns:
        list of lists: Filtered image (2D array).
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
    This version leverages the known structure of the Laplacian kernel to optimize the computation.
    
    Parameters:
        image_matrix (list of lists): Input image to be filtered (2D array).
    
    Returns:
        list of lists: Filtered image (2D array).
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
                + 4 * image_matrix[p][q] # Center
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
        image_matrix (numpy.ndarray): Input image matrix (2D array).
    
    Returns:
        numpy.ndarray: Filtered image matrix (2D array).
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
        matrix = imagineLoader(image_path)
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
            matrix_f = imagineLoader(output_path)
            print(mse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--pmse':
        if len(sys.argv) != 4:
            print("Usage: python script.py --pmse <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imagineLoader(output_path)
            print(pmse(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--snr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --snr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imagineLoader(output_path)
            print(snr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)
    
    elif command =='--psnr':
        if len(sys.argv) != 4:
            print("Usage: python script.py --psnr <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imagineLoader(output_path)
            print(psnr(matrix, matrix_f))
        except FileNotFoundError:
            print(f"Error: File {image_path} not found.")
            sys.exit(1)

    elif command =='--md':
        if len(sys.argv) != 4:
            print("Usage: python script.py --md <image_path> <image2_path>")
            sys.exit(1)
        try:
            matrix_f = imagineLoader(output_path)
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
            matrix_o = imagineLoader(origin_path)
            matrix_f = imagineLoader(output_path)
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
                print("Usage: python script.py --alpha <image_path> <gmin> <gmax> <alpha> <output_path>")
                sys.exit(1)
        try:
            g_min = int(sys.argv[3])
            g_max = int(sys.argv[4])
            alpha = float(sys.argv[5])
            if g_min < 0 or g_max < 0:
                print("G values must be bigger than 0")
                sys.exit(1)
            if (alpha < 0):
                print("Alpha must be between 0 and 0.5.")
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
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

