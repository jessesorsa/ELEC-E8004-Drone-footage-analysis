import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

f = 2.8
K = 8
t = 1/6.25
S = 400


# vignetting calibration image path
vignetting_image_path = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/vignetting/images/vignetting.jpg')

# photo_5809671548720758861_y
image_path_ = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/vignetting/images/calibration/photo_5809671548720758867_y.jpg')

# image path (path to image that has a center with 150 luminance)
image_path_150 = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/vignetting/images/calibration/photo_5809671548720758865_y.jpg')

# image paths to known luminancs (1 etc is the known luminance 1)
path_1_25 = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/pictures/jpeg/CCL1,250.jpg')
path_2 = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/pictures/jpeg/CCL2,000.jpg')
path_3 = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/pictures/jpeg/CCL3,000.jpg')
path_4 = os.path.join(
    '/Users/jessesorsa/Koulu/Urban_lighting_project/project/pictures/jpeg/CCL4,000.jpg')


# calculate pixel luminance
def luminance_equation(r, g, b):
    Y = (0.2126*r)+(0.7152*g)+(0.0722*b)
    L = (Y*np.power(f, 2))/(K*t*S)
    return L

# determine calibration constant


def calculate_K(r, g, b, L):
    Y = (0.2126*r)+(0.7152*g)+(0.0722*b)
    K = (Y*np.power(f, 2))/(L*t*S)
    return K


# importing imagex
def import_img(path):
    img = cv2.imread(path)
    return img

# printing image


def print_img(img):
    img = np.uint8(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def plotting_xy(x, y):
    # Plot the points
    plt.plot(x, y, color='red', marker='o', label='Points')

    # Set the axis limits
    plt.xlim(1, 4.5)
    plt.ylim(1, 4.5)

    # Add labels and title
    plt.xlabel('Measured luminances')
    plt.ylabel('Calculated luminances')
    plt.title('Measured vs calculated luminance')

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


# printing the image pixel luminance values between 0-255
def print_luminance(luminance_array):
    array = np.zeros_like(luminance_array, dtype=np.float32)
    height = luminance_array.shape[0]
    width = luminance_array.shape[1]

    max_value = np.max(luminance_array)

    t0 = max_value
    t1 = 3/4 * max_value
    t2 = 2/4 * max_value
    t3 = 1/4 * max_value

    # this scales the luminance values to be between 0-255
    scaling_constant = 255/max_value

    for y in range(height):
        for x in range(width):
            if (luminance_array[y, x][0] <= t3):
                array[y, x] = [63, 12, 144]
            if (t3 < luminance_array[y, x][0] <= t2):
                array[y, x] = [57, 0, 199]
            if (t2 < luminance_array[y, x][0] <= t1):
                array[y, x] = [16, 76, 249]
            if (t1 < luminance_array[y, x][0] <= t0):
                array[y, x] = [34, 222, 248]

            # array[y, x] = scaling_constant * luminance_array[y, x]

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

            luminance_array[y, x] = [luminance_value]
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


# Testing luminance calculations
img_1 = import_img(path_1_25)
img_2 = import_img(path_2)
img_3 = import_img(path_3)
img_4 = import_img(path_4)

crop_1 = middle_image_crop(img_1, 20)
crop_2 = middle_image_crop(img_2, 20)
crop_3 = middle_image_crop(img_3, 20)
crop_4 = middle_image_crop(img_4, 20)

print_img(crop_1)
print_img(crop_2)
print_img(crop_3)
print_img(crop_4)

luminance_1 = luminance(crop_1)
luminance_2 = luminance(crop_2)
luminance_3 = luminance(crop_3)
luminance_4 = luminance(crop_4)

mean_luminance_1 = luminance_mean(luminance_1)
mean_luminance_2 = luminance_mean(luminance_2)
mean_luminance_3 = luminance_mean(luminance_3)
mean_luminance_4 = luminance_mean(luminance_4)

L_calculated_array = [mean_luminance_1, mean_luminance_2,
                      mean_luminance_3, mean_luminance_4]

"""
# These should be around: 1,25, 2, 3, and 4, if calibration constant is correct
print(mean_luminance_1)
print(mean_luminance_2)
print(mean_luminance_3)
print(mean_luminance_4)
"""

K_array = np.zeros(4)

b1, g1, r1 = crop_1[10, 10]
b2, g2, r2 = crop_2[10, 10]
b3, g3, r3 = crop_3[10, 10]
b4, g4, r4 = crop_4[10, 10]

K_array[0] = calculate_K(r1, g1, b1, 1.25)
K_array[1] = calculate_K(r2, g2, b2, 2)
K_array[2] = calculate_K(r3, g3, b3, 3)
K_array[3] = calculate_K(r4, g4, b4, 4)

print(K_array)

L_array = [1.25, 2, 3, 4]

plotting_xy(L_array, L_calculated_array)

"""

##
##
##
##
##
##
# Correcting vignetting / This is the actual vignetting correction part

vignetting_img = import_img(vignetting_image_path)
image_150 = import_img(image_path_150)
resized_image_150 = resize_image(
    image_150, vignetting_img.shape[0], vignetting_img.shape[1])
luminance_array1 = luminance(resized_image_150)
luminance_array2 = luminance(vignetting_img)

corrected_luminance1 = vignetting_correction(luminance_array1, vignetting_img)

print_luminance(corrected_luminance1)
"""
