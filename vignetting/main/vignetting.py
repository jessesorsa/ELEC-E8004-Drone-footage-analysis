import cv2
import os
import numpy as np

f = 2.8
K = 0.9
t = 1/60
S = 200


# vignetting calibration image path
calibration_image_path = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/vignetting/images/vignetting.jpg')

# image path
image_path = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/vignetting/images/Baltimore_Oriole-Matthew_Plante.jpg')


# calculate pixel luminance
def luminance_equation(r, g, b):
    Y = (0.2126*r)+(0.7152*g)+(0.0722*b)
    L = (Y*np.power(f, 2))/(K*t*S)
    return L

# importing image


def import_img(path):
    img = cv2.imread(path)
    return img

# printing image


def print_img(img):
    img = np.uint8(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

# printing the image pixel luminance values between 0-255


def print_luminance(luminance_array):
    array = np.zeros_like(luminance_array, dtype=np.float32)
    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    max_value = np.max(luminance_array)

    # this scales the luminance values to be between 0-255
    scaling_constant = 255/max_value

    for y in range(height):
        for x in range(width):
            array[y, x] = scaling_constant * luminance_array[y, x]

    array = np.uint8(array)
    cv2.imshow('image', array)
    cv2.waitKey(0)

# resizing image


def resize_image(img, height, width):
    resized_img = cv2.resize(img, (width, height))
    return resized_img


# Taking the middle of an image
def middle_image_crop(img, size):
    # Img is the original image and size is the wanted size (size x size) of the middle square
    # get shape
    height = img.shape[0]
    width = img.shape[1]

    # get middle point
    y_middle = height/2
    x_middle = width/2

    # crop the image based on size
    y_lower_limit = int(y_middle - (size/2))
    y_upper_limit = int(y_middle + (size/2))
    x_lower_limit = int(x_middle-(size/2))
    x_upper_limit = int(x_middle+(size/2))
    cropped_image = img[y_lower_limit:y_upper_limit,
                        x_lower_limit:x_upper_limit]
    return cropped_image

# calucluating mean luminance from luminance array


def luminance_mean(luminance_array):
    height = luminance_array.shape[0]
    width = luminance_array.shape[1]
    number_of_pixels = height * width
    sum = 0
    for y in range(height):
        for x in range(width):
            sum += luminance_array[y, x]
    mean = sum / number_of_pixels
    return mean

# calculate the luminance of an image


def luminance(img):
    luminance_array = np.zeros_like(img, dtype=np.float32)
    height = img.shape[0]
    width = img.shape[1]

    for y in range(height):
        for x in range(width):
            b, g, r = (img[y, x])

            luminance_value = luminance_equation(r, g, b)

            luminance_array[y, x] = [luminance_value,
                                     luminance_value, luminance_value]
    return luminance_array

# creating the vignetting correction matrix based on a calibration image


def create_vignetting_correction_matrix(cal_img):

    luminance_array = luminance(cal_img)
    vignetting_correction_matrix = np.zeros_like(
        luminance_array, dtype=np.float32)

    middle_image = middle_image_crop(luminance_array, 50)
    middle_mean = luminance_mean(middle_image)

    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    for y in range(height):
        for x in range(width):
            difference = middle_mean - luminance_array[y, x]
            vignetting_correction_matrix[y, x] = difference
    return vignetting_correction_matrix


# operate the vignetting correction on a luminance matrix, based on a calibration image
def vignetting_correction(luminance_array, calibration_img):

    # This would actually be replaced with the static correction matrix that is calculated
    # For now it is always calculated from the calibration_image
    correction_matrix = create_vignetting_correction_matrix(calibration_img)
    luminance = (luminance_array + correction_matrix)
    return luminance


calibration_img = import_img(calibration_image_path)
image = import_img(image_path)

resized_image = resize_image(
    image, calibration_img.shape[0], calibration_img.shape[1])

print_img(resized_image)

luminance_array1 = luminance(resized_image)
luminance_array2 = luminance(calibration_img)

print_luminance(luminance_array1)

corrected_luminance1 = vignetting_correction(luminance_array1, calibration_img)
corrected_luminance2 = vignetting_correction(luminance_array2, calibration_img)

print_luminance(corrected_luminance1)

print_luminance(luminance_array2)
print_luminance(corrected_luminance2)
