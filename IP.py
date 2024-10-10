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
    if len(image_matrix.shape) == 2:  
        # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                if sign_negative == False:
                    # Increase brightness
                    if (255 - image_matrix[i, j] <= brightness_change):
                        output_matrix[i, j] = 255
                    else:
                        output_matrix[i, j] = image_matrix[i, j] + brightness_change     
                else:
                    # Decrease brightness
                    if (image_matrix[i, j] <= brightness_change):
                        output_matrix[i, j] = 0
                    else:
                        output_matrix[i, j] = image_matrix[i, j] - brightness_change
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    if sign_negative == False:
                        # Increase brightness
                        if (255 - image_matrix[i, j, k] <= brightness_change):
                            output_matrix[i, j, k] = 255
                        else:
                            output_matrix[i, j, k] = image_matrix[i, j, k] + brightness_change     
                    else:
                        # Decrease brightness
                        if (image_matrix[i, j, k] <= brightness_change):
                            output_matrix[i, j, k] = 0
                        else:
                            output_matrix[i, j, k] = image_matrix[i, j, k] - brightness_change
    return output_matrix

def brightnessChangerFlatLT(image_matrix, brightness_change, sign_negative):
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



def brightnessChangerGammaLT(image_matrix, brightness_change):
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
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                # Figure out clipping
                # co jeśli w (image_matrix[i, j] - 128), image_matrix[i, j] < 128?? Clipping baby
                if(image_matrix[i, j] == 0):
                    output_matrix[i,j] = 0
                elif (contrast_factor * (int(image_matrix[i, j]) - 128) + 128 > 255):
                    output_matrix[i,j] = 255
                elif (contrast_factor * (int(image_matrix[i, j]) - 128) + 128 < 0):
                    output_matrix[i,j] = 0
                else:
                    output_matrix[i,j] = contrast_factor * (int(image_matrix[i, j]) - 128) + 128
                '''if (contrast_factor > (127 / (image_matrix[i, j] - 128))):
                   output_matrix[i,j] = 255
                elif (contrast_factor < (- 128 / (image_matrix[i, j] - 128))):
                    output_matrix[i,j] = 0
                else:
                    output_matrix[i,j] = contrast_factor * (image_matrix[i, j] - 128) + 128'''
                
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    # Figure out clipping
                    if(image_matrix[i, j, k] == 0):
                        output_matrix[i,j,k] = 0
                    elif (contrast_factor * (int(image_matrix[i, j, k]) - 128) + 128 > 255):
                        output_matrix[i,j,k] = 255
                    elif (contrast_factor * (int(image_matrix[i, j, k]) - 128) + 128 < 0):
                        output_matrix[i,j,k] = 0
                    else:
                        output_matrix[i,j,k] = contrast_factor * (int(image_matrix[i, j, k]) - 128) + 128
                    '''if (contrast_factor > (127 / (output_matrix[i,j,k] - 128))):
                        output_matrix[i,j,k] = 255
                    elif (contrast_factor < (- 128 / (output_matrix[i,j,k] - 128))):
                        output_matrix[i,j,k] = 0
                    else:
                        output_matrix[i,j,k] = contrast_factor * (output_matrix[i,j,k] - 128) + 128'''
                    
    return output_matrix

def contrastChangerLT(image_matrix, contrast_factor):
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

def alpha_trimmed_mean_filter(image_matrix, kernel_size=3, alpha=0.2):
    """
    Applies an alpha-trimmed mean filter to an image.

    Parameters:
    - image_matrix: Input image as a NumPy array (grayscale or RGB).
    - kernel_size: Size of the kernel (must be odd).
    - alpha: Fraction of lowest and highest values to discard (0 <= alpha < 0.5).

    Returns:
    - filtered_image: Image after applying the alpha-trimmed mean filter.
    """
    if alpha < 0 or alpha >= 0.5:
        raise ValueError("Alpha must be between 0 and 0.5")

    # Padding to handle the borders of the image
    pad_size = kernel_size // 2
    padded_image = np.pad(image_matrix, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'reflect')

    height, width, channels = image_matrix.shape
    filtered_image = np.zeros_like(image_matrix)

    # Number of pixels to remove on each end
    d = int(alpha * kernel_size * kernel_size)

    # Loop through the image
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                # Extract the kernel window
                window = padded_image[i:i+kernel_size, j:j+kernel_size, k].flatten()

                # Sort the window values
                sorted_window = np.sort(window)

                # Trim the d smallest and d largest values
                trimmed_window = sorted_window[d:-d]

                # Compute the mean of the remaining values
                filtered_image[i, j, k] = np.mean(trimmed_window)

    return filtered_image.astype(np.uint8)


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

    elif command == '--brightnessFlatLT':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessFlat <image_path> <brightness_value> <output_path>")
            sys.exit(1)
        try:
            brightness_value = int(sys.argv[3])
            sign_negative = brightness_value < 0  # Determine if brightness is increasing or decreasing
            modified_matrix = brightnessChangerFlatLT(matrix, abs(brightness_value), sign_negative)
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

    elif command == '--brightnessGammaLT':
        if len(sys.argv) != 5:
            print("Usage: python script.py --brightnessGamma <image_path> <brightness_factor> <output_path>")
            sys.exit(1)
        try:
            brightness_factor = float(sys.argv[3])
            modified_matrix = brightnessChangerGammaLT(matrix, brightness_factor)
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
        
    elif command == '--contrastLT':
        if len(sys.argv) != 5:
            print("Usage: python script.py --contrast <image_path> <contrast_factor> <output_path>")
            sys.exit(1)
        try:
            contrast_factor = float(sys.argv[3])
            modified_matrix = contrastChangerLT(matrix, contrast_factor)
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
            print("Error: Contrast factor must be a valid float.")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

