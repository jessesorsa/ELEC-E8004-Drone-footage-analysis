import cv2
import os
import numpy as np
from PIL import Image

# to the left of the tape the luminance should be 1.25

################# VARIABLES TO BE USED #####################
# N_d: Digital number (value) of the pixel in the image
# t: the camera exposure time in seconds
# f_s: Aperture number (f-stop)
# S: ISO Sensitivity of the film
# L_s: Luminance of the scene, candela/meter^2

def calculate_calibration_constant(f_s, N_d, t, S, L_s):
    # a function for calculating the calibration constant
    K_c = (f_s**2 * N_d)/(t*S*L_s)
    return K_c

def calculate_luminance(f_s, N_d, t, S, K_c):
    # a function for calculating the luminance
    L_s = (f_s**2 * N_d)/(K_c*t*S)
    return L_s

def calculate_luminance_for_pixel(pixel, f_s, t, S, K_c):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]

    if (r == 255 and g == 0 and b == 0):
        return 1000 # a distinct luminance value
    else:
        N_d = calculate_digital_number(pixel[0], pixel[1], pixel[2])
        return calculate_luminance(f_s, N_d, t, S, K_c)

def calculate_digital_number(R, G, B):
    # a function for calculating N_d value from RGB-values
    N_d = 0.2126*R + 0.7152*G + 0.0722*B
    return N_d

def calculate_middle_pixel_Nd(image):
    # a function for calculating N_d value for center pixel from image

    # accessing the center pixel
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    center_pixel = image[center_y][center_x]

    # obtaining RGB-values for pixel
    R = center_pixel[0]
    G = center_pixel[1]
    B = center_pixel[2]

    return calculate_digital_number(R, G, B)

def calculate_median_Nd(image):
    # function for calculating the N_d value of the median of the middle 100x100 pixels

    # accessing the 100x100 pixels in the middle
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    center_patch = image[center_y - 5:center_y + 5, center_x - 5:center_x + 5]

    # display the 100x100 pixel patch
    """ cv2.imshow('Center patch', center_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    # Flatten the region into a list of RGB values
    pixels = center_patch.reshape(-1, 3)

    # Calculate the median value for each color channel
    median_values = np.median(pixels, axis=0)

    # Convert the median values to integers
    median_values = np.uint8(median_values)

    # obtaining RGB-values for pixel
    R = median_values[0]
    G = median_values[1]
    B = median_values[2]

    return calculate_digital_number(R, G, B)

def calculate_luminances_for_image(ref_image, f_s, t, S, L_s):

    height = ref_image.shape[0]
    width = ref_image.shape[1]

    median_Nd = calculate_median_Nd(ref_image)

    # determine the calibration constant
    Kc_median = calculate_calibration_constant(f_s, median_Nd, t, S, L_s)

    luminances = np.zeros_like(ref_image, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            pixel = ref_image[y][x]
            luminances[y][x] = calculate_luminance_for_pixel(pixel, f_s, t, S, Kc_median)

    return luminances

# printing the image pixel luminance values between 0-255
def print_luminance(luminance_array):
    
    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    array = np.zeros((height, width, 3), dtype=np.uint8)

    max_value = np.max(luminance_array)

    t0 = max_value
    t1 = 3/4 * max_value
    t2 = 2/4 * max_value
    t3 = 1/4 * max_value

    # this scales the luminance values to be between 0-255
    scaling_constant = 255/max_value

    for y in range(height):
        for x in range(width):
            if (luminance_array[y][x][0] <= t3):
                array[y][x] = [63, 12, 144]
            if (t3 < luminance_array[y][x][0] <= t2):
                array[y][x] = [57, 0, 199]
            if (t2 < luminance_array[y][x][0] <= t1):
                array[y][x] = [16, 76, 249]
            if (t1 < luminance_array[y][x][0] <= t0):
                array[y][x] = [34, 222, 248]

            #array[y, x] = scaling_constant * luminance_array[y, x]

    array = np.uint8(array)
    """ cv2.imshow('image', array)
    cv2.waitKey(0) """

    return array


def save_image(array, exif, filename):

    # Create a folder path where you want to save the image
    folder_path = 'C:/Users/laura/Documents/AALTO/Project work/project_work_git/urban-sense/luminance-calculation/results'
    result_filename = filename + '_result.jpg'

    # Ensure that the folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    try:
        # Create an image from the array
        image = Image.fromarray(array)
        
        # Save the image
        image.save(os.path.join(folder_path, result_filename), exif=exif)
        print('Image saved successfully.')
    except Exception as e:
        print('Error saving image:', e)

def main():

    # Camera variables
    t = 1/1.625
    f_s = 2.8
    S = 400
    L_s = 0.262          # Adjust to the known luminance (candela/meter^2)

    # Get the current working directory
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Construct the relative path to the images folder
    relative_folder_path = "images/test_sample"

    # Combine the current directory with the relative folder path to get the absolute folder path
    image_folder_path = os.path.join(current_directory, relative_folder_path)

    # Get a list of all files in the directory
    file_list = os.listdir(image_folder_path)

    for file in file_list:
        image_filename = file

        # Load the image
        path = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(path, 'images/test_sample', image_filename)
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Unable to load image. Please check the file path.")
        else:

            # Convert the image to 16-bit depth
            image_float = image.astype(np.float32)
            image_16bit = (image_float / 255.0) * 65535.0
            image_16bit = np.clip(image_16bit, 0, 65535).astype(np.uint16)

            im = Image.open(image_path)
            exif = im.info['exif']

            filename_for_saving = '.'.join(image_filename.split('.')[:-1])

            luminances = calculate_luminances_for_image(image, f_s, t, S, L_s)
            new_image = print_luminance(luminances)
            save_image(new_image, exif, filename_for_saving)

if __name__ == '__main__':
    main()
    
    