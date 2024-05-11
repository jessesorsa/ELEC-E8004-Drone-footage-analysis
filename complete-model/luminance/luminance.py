import cv2
import os
import numpy as np
from PIL import Image

# Camera variables
t = 1/15
f_s = 2.8
S = 1600         # Adjust to the known luminance (candela/meter^2)


# to the left of the tape the luminance should be 1.25

################# VARIABLES TO BE USED #####################
# N_d: Digital number (value) of the pixel in the image
# t: the camera exposure time in seconds
# f_s: Aperture number (f-stop)
# S: ISO Sensitivity of the film
# L_s: Luminance of the scene, candela/meter^2


def calculate_calibration_constant(N_d, L_s):
    # a function for calculating the calibration constant
    K_c = (f_s**2 * N_d)/(t*S*L_s)
    return K_c


def calculate_luminance(N_d, K_c):
    # a function for calculating the luminance
    L_s = (f_s**2 * N_d)/(K_c*t*S)
    return L_s


def calculate_luminance_for_pixel(pixel, K_c):

    r = pixel[0]
    g = pixel[1]
    b = pixel[2]

    N_d = calculate_digital_number(r, g, b)
    return calculate_luminance(N_d, K_c)


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


def calculate_luminances_for_image(ref_image, constant):

    height = ref_image.shape[0]
    width = ref_image.shape[1]

    luminances = np.zeros_like(ref_image, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            pixel = ref_image[y][x]
            luminances[y][x] = calculate_luminance_for_pixel(
                pixel, constant)
    print("Returning luminance")
    return luminances

# printing the image pixel luminance values between 0-255


def print_luminance(luminance_array):

    print("Printing luminance")

    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    array = np.zeros((height, width, 3), dtype=np.uint8)

    max_value = np.max(luminance_array)

    t0 = max_value
    t1 = 12/20 * max_value
    t2 = 10/20 * max_value
    t3 = 8/20 * max_value
    t4 = 6/20 * max_value
    t5 = 3/20 * max_value
    t6 = 1/20 * max_value
    t7 = 1/30 * max_value
    t8 = 1/50 * max_value
    t9 = 0

    # this scales the luminance values to be between 0-255
    scaling_constant = 255/max_value

    for y in range(height):
        for x in range(width):
            if (luminance_array[y, x][0] == t9):
                array[y, x] = [0, 0, 0]
            if (t9 < luminance_array[y, x][0] <= t8):
                array[y, x] = [19, 42, 144]
            if (t8 < luminance_array[y, x][0] <= t7):
                array[y, x] = [110, 149, 255]
            if (t7 < luminance_array[y, x][0] <= t6):
                array[y, x] = [110, 236, 255]
            if (t6 < luminance_array[y, x][0] <= t5):
                array[y, x] = [110, 255, 187]
            if (t5 < luminance_array[y, x][0] <= t4):
                array[y, x] = [120, 255, 110]
            if (t4 < luminance_array[y, x][0] <= t3):
                array[y, x] = [207, 255, 110]
            if (t3 < luminance_array[y, x][0] <= t2):
                array[y, x] = [255, 216, 110]
            if (t2 < luminance_array[y, x][0] <= t1):
                array[y, x] = [255, 129, 110]
            if (t1 < luminance_array[y, x][0] <= t0):
                array[y, x] = [255, 255, 255]

            #array[y, x] = scaling_constant * luminance_array[y, x]

    array = np.uint8(array)
    """
    cv2.imshow('image', array)
    cv2.waitKey(0)
    """
    return array


def save_image(array, exif, filename):

    print("Saving image")

    # Create a folder path where you want to save the image
    folder_path = os.path.dirname(
        os.path.realpath(__file__)) + '/results'
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


def save_image_2(array, filename):

    print("Saving image")

    # Create a folder path where you want to save the image
    folder_path = os.path.dirname(
        os.path.realpath(__file__)) + '/results'
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


def resize_image(img, height, width):
    resized_img = cv2.resize(img, (width, height))
    return resized_img


def calibration_with_image():

    # this method is specific to the picture below

    path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(
        path, 'images/calibration_image', 'DJI_20240428234322_0004_V.JPG')
    image = cv2.imread(image_path)

    # accessing the pixels of the white square in the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    white_patch = image[center_y + 90:center_y +
                        120, center_x + 110:center_x + 140]

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


def main(scaling_factor):

    constant = calibration_with_image()

    # Get the current working directory
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Construct the relative path to the images folder
    relative_folder_path = "../ML_model/images"

    # Combine the current directory with the relative folder path to get the absolute folder path
    image_folder_path = os.path.join(current_directory, relative_folder_path)

    # Get a list of all files in the directory
    file_list = os.listdir(image_folder_path)

    for file in file_list:
        image_filename = file

        # Load the image
        image_path = image_folder_path + "/" + image_filename
        original_image = cv2.imread(image_path)

        height, width, rgb = original_image.shape
        image = resize_image(original_image, int(
            height/scaling_factor), int(width/scaling_factor))

        if image is None:
            print("Error: Unable to load image. Please check the file path.")
        else:
            """
            # Convert the image to 16-bit depth
            image_float = image.astype(np.float32)
            image_16bit = (image_float / 255.0) * 65535.0
            image_16bit = np.clip(image_16bit, 0, 65535).astype(np.uint16)
            """
            im = Image.open(image_path)

            filename_for_saving = '.'.join(image_filename.split('.')[:-1])

            luminances = calculate_luminances_for_image(
                image, constant)
            new_image = print_luminance(luminances)
            if (im.info.get("exif")):
                save_image(new_image, im.info.get("exif"), filename_for_saving)
            else:
                save_image_2(new_image, filename_for_saving)


if __name__ == '__main__':
    main()
