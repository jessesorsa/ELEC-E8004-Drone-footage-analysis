import cv2
import os
import numpy as np
from PIL import Image

# Camera variables
t = 1/15
f_s = 2.8
s = 1600

# to the left of the tape the luminance should be 1.25

################# VARIABLES TO BE USED #####################
# N_d: Digital number (value) of the pixel in the image
# t: the camera exposure time in seconds
# f_s: Aperture number (f-stop)
# S: ISO Sensitivity of the film
# L_s: Luminance of the scene, candela/meter^2

def calculate_calibration_constant(N_d, L_s):
    # a function for calculating the calibration constant
    K_c = (f_s**2 * N_d)/(t*s*L_s)
    return K_c

def calculate_luminance(N_d, K_c):
    # a function for calculating the luminance
    L_s = (f_s**2 * N_d)/(K_c*t*s)
    return L_s

def calculate_luminance_for_pixel(pixel, K_c):
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]

    if (r == 0 and g == 0 and b == 0):
        return 0 # a distinct luminance value
    else:
        N_d = calculate_digital_number(pixel[0], pixel[1], pixel[2])
        return calculate_luminance(N_d, K_c)

def calculate_digital_number(r, g, b):
    # a function for calculating N_d value from RGB-values
    N_d = 0.2126*r + 0.7152*g + 0.0722*b
    return N_d

def calculate_middle_pixel_Nd(image):
    # a function for calculating N_d value for center pixel from image

    # accessing the center pixel
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    center_pixel = image[center_y][center_x]

    # obtaining RGB-values for pixel
    r = center_pixel[0]
    g = center_pixel[1]
    b = center_pixel[2]

    return calculate_digital_number(r, g, b)

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

def calculate_luminances_for_image(ref_image, calibration_constant):

    height = ref_image.shape[0]
    width = ref_image.shape[1]

    luminances = np.zeros_like(ref_image, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            pixel = ref_image[y][x]
            luminances[y][x] = calculate_luminance_for_pixel(pixel, calibration_constant)

    return luminances

def print_luminance(luminance_array):
    
    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    array = np.zeros((height, width, 3), dtype=np.uint8)

    max_value = np.max(luminance_array)

    t0 = max_value
    t1 = 3/4 * max_value
    t2 = 2/4 * max_value
    t3 = 1/4 * max_value

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
            # if the pixel is completely black
            if (np.array_equal(luminance_array[y][x], [0, 0, 0])):
                array[y][x] = [0, 0, 0]

    array = np.uint8(array)
    # uncomment if want to display image
    """ cv2.imshow('image', array)
    cv2.waitKey(0) """

    return array

def calibration_with_image():

    # this method is specific to the picture below

    path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(path, 'images/calibration_image', 'DJI_20240428234322_0004_V.JPG')
    image = cv2.imread(image_path)

    # accessing the pixels of the white square in the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    white_patch = image[center_y + 90:center_y + 120, center_x + 110:center_x + 140]

    # uncomment if want to display the pixel patch
    """ cv2.imshow('Center patch', white_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    # Flatten the region into a list of RGB values
    pixels = white_patch.reshape(-1, 3)

    # Calculate the median value for each color channel
    median_values = np.median(pixels, axis=0)

    # Convert the median values to integers
    median_values = np.uint8(median_values)

    # obtaining RGB-values for pixel
    r = median_values[0]
    g = median_values[1]
    b = median_values[2]

    known_luminance = 1.6       # from known luminance measurements of the white square
    nd = calculate_digital_number(r, g, b)
    constant = calculate_calibration_constant(nd, known_luminance)

    return constant


def save_image_with_exif(array, exif, filename):

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

def save_image(array, filename):
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
        image.save(os.path.join(folder_path, result_filename))
        print('Image saved successfully.')

    except Exception as e:
        print('Error saving image:', e)

def resize_image(image, width):

    # Calculate the scaling factor for resizing
    scale_factor = width / image.shape[1]

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    return resized_image

def plot_one(filename):

    constant = calibration_with_image()

    # Load the image
    path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(path, 'images', filename)
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image. Please check the file path.")
    else:

        # Convert the image to 16-bit depth
        image_float = image.astype(np.float32)
        image_16bit = (image_float / 255.0) * 65535.0
        image_16bit = np.clip(image_16bit, 0, 65535).astype(np.uint16)

        filename_for_saving = '.'.join(filename.split('.')[:-1]) # original filename with _results.jpg at the end

        luminances = calculate_luminances_for_image(image, constant)
        new_image = print_luminance(luminances)
        save_image(new_image, filename_for_saving)

def plot_all():

    constant = calibration_with_image()

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

            luminances = calculate_luminances_for_image(image, constant)
            new_image = print_luminance(luminances)
            save_image_with_exif(new_image, exif, filename_for_saving)

def main():
    #plot_all()
    plot_one('Testsample20240422.jpg')

if __name__ == '__main__':
    main()
    
    