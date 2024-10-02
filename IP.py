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

def brightnessChangerFlat(image_matrix, brightness_change, sign_negative):
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    if len(image_matrix.shape) == 2:  # Grayscale image
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

def brightnessChangerGamma(image_matrix, brightness_change):
    if(brightness_change < 0):
        print("Wrong brightness_changer!")
        return image_matrix
    output_matrix = np.zeros_like(image_matrix, dtype=np.uint8)
    if len(image_matrix.shape) == 2:  # Grayscale image
        height, width = image_matrix.shape
        for i in range(height):
            for j in range(width):
                if (brightness_change > 255/image_matrix[i, j]):
                    output_matrix[i, j] = 255
                else:
                    output_matrix[i, j] = image_matrix[i, j] * brightness_change
    else:
        # RGB Image
        height, width, channels = image_matrix.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    if (brightness_change > 255/image_matrix[i, j, k]):
                        output_matrix[i, j, k] = 255
                    else:
                        output_matrix[i, j, k] = image_matrix[i, j, k] * brightness_change   
    return output_matrix

def saveImage(image_matrix, output_path):
    new_image = Image.fromarray(image_matrix)
    new_image.save(output_path)
    print(f"Image saved at {output_path}")

### CMD commands
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <command> <image_path> [<parameters>]")
        sys.exit(1)

    command = sys.argv[1]
    image_path = sys.argv[2]

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
            output_path = sys.argv[4]
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
            output_path = sys.argv[4]
            modified_matrix = brightnessChangerGamma(matrix, brightness_factor)
            saveImage(modified_matrix, output_path)
        except ValueError:
            print("Error: Brightness factor must be a valid float.")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
