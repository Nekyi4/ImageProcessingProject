from PIL import Image
import numpy as np
import sys

### Image processing

def imagineLoader(param):
    im = Image.open(param)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image_arr = np.array(im, dtype=np.uint8)
    return image_arr

def saveImage(image_matrix, output_path):
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
    d = max(1, int(alpha * kernel_size * kernel_size))
    border = int((kernel_size - 1) / 2) 

    for i in range(border, height - border):
        for j in range(border, width - border):
            for k in range(channels):
                window =  window = image_matrix[i - border:i + border+1, j - border:j + border+1, k].flatten()
                sorted_window = np.sort(window)
                # Check if d is less than the remaining number of pixels after trimming
                if d < len(sorted_window) // 2:
                    trimmed_window = sorted_window[d:-d]
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
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

