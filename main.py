from __future__ import print_function
import argparse
import numpy as np
import torch
import random
import os
import pathlib
import random
from PIL import Image
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes # for categorizing training/testing images
from nuscenes.utils import splits

# get the low frequency using Gaussian Low-Pass Filter
# high d value tends to make high frequency info empty, low frequency info less blurred (intensifies high filter, weakens low filter)
# in other words, high d value increases threshold for what counts as high frequency information rather than low
def gaussian_filter_low_pass(fshift, D, perturb=False):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to dis_square
    if perturb:
        real_noise_factor = np.random.normal(0.5, 0.5, dis_square.shape)
        imaginary_noise_factor = np.random.normal(0.5, 0.5, dis_square.shape)
        fshift.real = fshift.real * real_noise_factor
        fshift.imag = fshift.imag * imaginary_noise_factor
        #real_noise = np.random.normal(0, 99999, dis_square.shape)
        #imaginary_noise = np.random.normal(0, 99999, dis_square.shape)
        #fshift = fshift + real_noise + imaginary_noise * 1j
    return template * fshift

# get the high frequency using Gaussian High-Pass Filter
def gaussian_filter_high_pass(fshift, D, perturb=False):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to 0
    # template becomes >0.5 a mere 12 pixels away from center! (with default D)
    #print(template.shape)
    #print(template[center[0], center[1] + 12])
    #print(template[center[0] + 12, center[1]])
    #assert 0
    if perturb:
        real_noise_factor = np.random.normal(0.5, 0.5, dis_square.shape)
        imaginary_noise_factor = np.random.normal(0.5, 0.5, dis_square.shape)
        fshift.real = fshift.real * real_noise_factor
        fshift.imag = fshift.imag * imaginary_noise_factor
        #real_noise = np.random.normal(0, 99999, dis_square.shape)
        #imaginary_noise = np.random.normal(0, 99999, dis_square.shape)
        #fshift = fshift + real_noise + imaginary_noise * 1j
    return template * fshift

def create_circular_mask(h, w, radius):
    center = (int(w/2), int(h/2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center > radius
    return mask

# Inverse Fourier transform
def ifft(fshift):
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifftn(ishift)
    iimg = np.abs(iimg)
    return iimg

def f_shift(img_c): # given one channel image
    f = np.fft.fftn(img_c)
    f_shift = np.fft.fftshift(f)
    return f_shift

def f_shift_rgb(img):
    img = np.array(img)
    f_shift_r = f_shift(img[:, :, 0])
    f_shift_g = f_shift(img[:, :, 1])
    f_shift_b = f_shift(img[:, :, 2])
    f_shift_rgb = np.array([f_shift_r, f_shift_g, f_shift_b])
    return f_shift_rgb

def magnitude_visualize(img):
    '''Convert image to grayscale and return visualization of magnitudes of transformed image.'''
    image_grayscale = img.convert('L')
    img_arr = np.array(image_grayscale)
    img_arr_f_shift = f_shift(img_arr)
    # Calculate the magnitude spectrum, scaled so that it is useful for visualization
    magnitude_spectrum = 14*np.log(np.abs(img_arr_f_shift))
    magnitude_image = Image.fromarray(magnitude_spectrum)
    return magnitude_image

def magnitude_visualize_raw(img):
    '''Convert image to grayscale and return visualization of magnitudes of transformed image
    as-is, with no log or multiplication for ease of visualization.
    Will probably just give a white pixel in the center.'''
    image_grayscale = img.convert('L')
    img_arr = np.array(image_grayscale)
    img_arr_f_shift = f_shift(img_arr)
    # Calculate the magnitude spectrum, scaled so that it is useful for visualization
    magnitudes = np.abs(img_arr_f_shift)
    max_magnitude = np.max(magnitudes)
    magnitude_spectrum = np.abs(img_arr_f_shift)/max_magnitude*255
    magnitude_image = Image.fromarray(magnitude_spectrum)
    return magnitude_image

def phase_visualize(img):
    '''Convert image to grayscale and return visualization of phase angles of transformed image.'''
    image_grayscale = img.convert('L')
    img_arr = np.array(image_grayscale)
    img_arr_f_shift = f_shift(img_arr)
    # Calculate the phase, scaled so that it is useful for visualization
    phase = 40*np.angle(img_arr_f_shift)
    phase_image = Image.fromarray(phase)
    return phase_image

def magnitude_visualize_rgb(img):
    img_f_shift_rgb = f_shift_rgb(img)
    h, w = img_f_shift_rgb[0].shape # doesn't matter which channel we use here
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = 14*np.log(np.abs(img_f_shift_rgb[0]))
    rgbArray[:, :, 1] = 14*np.log(np.abs(img_f_shift_rgb[1]))
    rgbArray[:, :, 2] = 14*np.log(np.abs(img_f_shift_rgb[2]))
    magnitude_image = Image.fromarray(rgbArray)
    return magnitude_image

def low_pass_rgb(img, D, perturb=False):
    img_f_shift_rgb = f_shift_rgb(img)
    low_part_r = gaussian_filter_low_pass(img_f_shift_rgb[0].copy(), D, perturb)
    low_part_g = gaussian_filter_low_pass(img_f_shift_rgb[1].copy(), D, perturb)
    low_part_b = gaussian_filter_low_pass(img_f_shift_rgb[2].copy(), D, perturb)
    low_rgb = np.array([low_part_r, low_part_g, low_part_b])
    return low_rgb

def high_pass_rgb(img, D, perturb=False):
    img_f_shift_rgb = f_shift_rgb(img)
    high_part_r = gaussian_filter_high_pass(img_f_shift_rgb[0].copy(), D, perturb)
    high_part_g = gaussian_filter_high_pass(img_f_shift_rgb[1].copy(), D, perturb)
    high_part_b = gaussian_filter_high_pass(img_f_shift_rgb[2].copy(), D, perturb)
    high_rgb = np.array([high_part_r, high_part_g, high_part_b])
    return high_rgb

def low_high_pass_rgb(img, D, perturb_low=False, perturb_high=False):
    low_rgb = low_pass_rgb(img, D, perturb_low)
    high_rgb = high_pass_rgb(img, D, perturb_high)
    low_high_rgb = np.array([low_rgb, high_rgb])
    return low_high_rgb

def hard_filter_rgb(img, radius):
    img_f_shift_rgb = f_shift_rgb(img)
    h, w = img_f_shift_rgb[0].shape
    mask = create_circular_mask(h, w, radius)
    img_f_shift_rgb[0] = img_f_shift_rgb[0] * mask
    img_f_shift_rgb[1] = img_f_shift_rgb[1] * mask
    img_f_shift_rgb[2] = img_f_shift_rgb[2] * mask
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = ifft(img_f_shift_rgb[0])
    rgbArray[:, :, 1] = ifft(img_f_shift_rgb[1])
    rgbArray[:, :, 2] = ifft(img_f_shift_rgb[2])
    new_img = Image.fromarray(rgbArray)
    return new_img

def augment_image(r_img, i_img, D, augment_high=True): # reference image, interference image
    channel_num = random.randrange(0,3) # Randomize which channel to interfere with - 0 is red, 1 is green, 2 is blue
    i_img_low_high_parts = low_high_pass_rgb(i_img, D)
    r_img_low_high_parts = low_high_pass_rgb(r_img, D)

    if augment_high:
        augmented_low_high_parts_r = r_img_low_high_parts[0][0] + (i_img_low_high_parts[1][0] if channel_num == 0 else r_img_low_high_parts[1][0])
        augmented_low_high_parts_g = r_img_low_high_parts[0][1] + (i_img_low_high_parts[1][1] if channel_num == 1 else r_img_low_high_parts[1][1])
        augmented_low_high_parts_b = r_img_low_high_parts[0][2] + (i_img_low_high_parts[1][2] if channel_num == 2 else r_img_low_high_parts[1][2])
    else:
        augmented_low_high_parts_r = r_img_low_high_parts[1][0] + (i_img_low_high_parts[0][0] if channel_num == 0 else r_img_low_high_parts[0][0])
        augmented_low_high_parts_g = r_img_low_high_parts[1][1] + (i_img_low_high_parts[0][1] if channel_num == 1 else r_img_low_high_parts[0][1])
        augmented_low_high_parts_b = r_img_low_high_parts[1][2] + (i_img_low_high_parts[0][2] if channel_num == 2 else r_img_low_high_parts[0][2])
    img_r = ifft(augmented_low_high_parts_r)
    img_g = ifft(augmented_low_high_parts_g)
    img_b = ifft(augmented_low_high_parts_b)
    h, w = augmented_low_high_parts_r.shape # doesn't matter which channel we use here
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = img_r
    rgbArray[:, :, 1] = img_g
    rgbArray[:, :, 2] = img_b
    img = Image.fromarray(rgbArray)
    return img

def enhance_high(img, D):
    '''Enhance high-frequency information and turn down low-frequency information.'''
    low_high_parts = low_high_pass_rgb(img, D)
    low_high_parts[0][0] *= 0.5
    low_high_parts[0][1] *= 0.5
    low_high_parts[0][2] *= 0.5
    low_high_parts[1][0] *= 1.5
    low_high_parts[1][1] *= 1.5
    low_high_parts[1][2] *= 1.5
    img_r = low_high_parts[0][0] + low_high_parts[1][0]
    img_g = low_high_parts[0][1] + low_high_parts[1][1]
    img_b = low_high_parts[0][2] + low_high_parts[1][2]
    h,w = img_r.shape
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = ifft(img_r)
    rgbArray[:, :, 1] = ifft(img_g)
    rgbArray[:, :, 2] = ifft(img_b)
    img = Image.fromarray(rgbArray)
    return img

def perturb_image(img, D, perturb_high=True):
    '''Perturb either high or low frequency information. Default high, otherwise low.'''
    if perturb_high:
        img_low_high_parts = low_high_pass_rgb(img, D, perturb_high=True)
    else:
        img_low_high_parts = low_high_pass_rgb(img, D, perturb_low=True)
    img_r = ifft(img_low_high_parts[0][0]+img_low_high_parts[1][0])
    img_g = ifft(img_low_high_parts[0][1]+img_low_high_parts[1][1])
    img_b = ifft(img_low_high_parts[0][2]+img_low_high_parts[1][2])
    h, w = img_low_high_parts[0][0].shape # doesn't matter which channel we use here
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = img_r
    rgbArray[:, :, 1] = img_g
    rgbArray[:, :, 2] = img_b
    img = Image.fromarray(rgbArray)
    return img

def shuffle_image(img, D, k, radius, restrict=False, shuffleMax=False):
    channel_num = random.randrange(0,3) # Randomize which channel to interfere with - 0 is red, 1 is green, 2 is blue
    img_low_high_parts = low_high_pass_rgb(img, D)

    target = img_low_high_parts[1][channel_num] # spectrum to shuffle
    magnitudes = np.abs(target) # 900 rows by 1600 columns for nuscenes cam image
    if restrict:
        # prevent use of strong low frequencies near origin
        mask = create_circular_mask(magnitudes.shape[0], magnitudes.shape[1], radius)
        magnitudes = magnitudes * mask
    if shuffleMax:
        target_indices_flattened = np.argpartition(magnitudes.flatten(), -1 * k)[-1 * k:]
    else:
        target_indices_flattened = np.argpartition(magnitudes.flatten(), k)[:k]
    target_indices = [np.unravel_index(i, magnitudes.shape) for i in target_indices_flattened]
    # prints for debugging
    #sorted_target_indices_flattened = target_indices_flattened[np.argsort(magnitudes.flatten()[target_indices_flattened])]
    #sorted_target_indices = [np.unravel_index(i, magnitudes.shape) for i in sorted_target_indices_flattened]
    #for index in sorted_target_indices:
        #print(index)
        #print(magnitudes[index[0], index[1]])

    new_indices = random.sample(target_indices, len(target_indices))
    for i, pixel in enumerate(target_indices):
        new = new_indices[i] # pixel to swap with
        new_y = new[0]
        new_x = new[1]
        old_y = pixel[0]
        old_x = pixel[1]
        temp = target[old_y, old_x]
        target[old_y, old_x] = target[new_y, new_x]
        target[new_y, new_x] = temp

    shuffled_low_high_parts_r = img_low_high_parts[0][0] + img_low_high_parts[1][0]
    shuffled_low_high_parts_g = img_low_high_parts[0][1] + img_low_high_parts[1][1]
    shuffled_low_high_parts_b = img_low_high_parts[0][2] + img_low_high_parts[1][2]

    #shuffled_low_high_parts_r[440:460, 790:810] = 0
    #shuffled_low_high_parts_g[440:460, 790:810] = 0
    #shuffled_low_high_parts_b[440:460, 790:810] = 0

    img_r = ifft(shuffled_low_high_parts_r)
    img_g = ifft(shuffled_low_high_parts_g)
    img_b = ifft(shuffled_low_high_parts_b)
    h, w = shuffled_low_high_parts_r.shape # doesn't matter which channel we use here
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = img_r
    rgbArray[:, :, 1] = img_g
    rgbArray[:, :, 2] = img_b
    img = Image.fromarray(rgbArray)
    return img

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FFT/iFFT test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--reference-image', type=str, metavar='filename',
                        help='reference image to transform or augment with interference image')
    parser.add_argument('--interference-image', type=str, metavar='interference filename',
                        help='interference image used to augment reference image')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--d', type=int, default=10,
                        help='d value in gaussian function')
    parser.add_argument('--k', type=int, default=10,
                        help='k value in shuffle function')
    parser.add_argument('--radius', type=int, default=100,
                        help='k value in shuffle function')
    parser.add_argument('--restrict', action='store_true', default=False,
                        help='blocks frequencies near the center in fourier'
                             'domain from being considered for shuffling')
    parser.add_argument('--shuffle-max', action='store_true', default=False,
                        help='shuffle frequencies of max magnitude')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #parser.add_argument('--no-mps', action='store_true', default=False,
    #                    help='disables macOS GPU training')
    parser.add_argument('--show-transforms', action='store_true', default=False,
                        help='show high and low pass transforms on each channel of image')
    parser.add_argument('--augment-image', action='store_true', default=False,
                        help='show image augmented in the same manner as the FDA paper')
    parser.add_argument('--magnitude-visualize', action='store_true', default=False,
                        help='show visualization of magnitudes of transformed image')
    parser.add_argument('--magnitude-visualize-raw', action='store_true', default=False,
                        help='show raw visualization of magnitudes of transformed image')
    parser.add_argument('--phase-visualize', action='store_true', default=False,
                        help='show visualization of phase angles of transformed image')
    parser.add_argument('--magnitude-visualize-rgb', action='store_true', default=False,
                        help='show visualization of magnitudes of transformed image in RGB')
    parser.add_argument('--high-freq', action='store_true', default=False,
                        help='get the high frequency components of the reference image in RGB')
    parser.add_argument('--low-freq', action='store_true', default=False,
                        help='get the low frequency components of the reference image in RGB')
    parser.add_argument('--perturb-low', action='store_true', default=False,
                        help='show low-frequency perturbation of reference image in RGB')
    parser.add_argument('--perturb-high', action='store_true', default=False,
                        help='show high-frequency perturbation of reference image in RGB')
    parser.add_argument('--enhance-high', action='store_true', default=False,
                        help='show reference image with high-frequency information enhanced over low-frequency info')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='show reference image with high-frequency information shuffled')
    parser.add_argument('--nuscenes-samples-preprocess', action='store_true', default=False,
                        help='save low and high frequency components of nuscenes samples to new directory "preproces"')
    parser.add_argument('--nuscenes-samples-augment-high', action='store_true', default=False,
                        help='save randomly high-frequency augmented versions of images')
    parser.add_argument('--nuscenes-samples-augment-low', action='store_true', default=False,
                        help='save randomly low-frequency augmented versions of images')
    parser.add_argument('--nuscenes-samples-perturb-high', action='store_true', default=False,
                        help='save randomly high-frequency perturbed versions of images')
    parser.add_argument('--nuscenes-samples-perturb-low', action='store_true', default=False,
                        help='save randomly low-frequency perturbed versions of images')
    parser.add_argument('--nuscenes-samples-enhance-high', action='store_true', default=False,
                        help='save high frequency enhanced versions of images')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    #elif use_mps
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # FFT stuff
    if args.reference_image is not None:
        image = Image.open(args.reference_image)
        if args.augment_image:
            if args.interference_image is not None:
                interference_image = Image.open(args.interference_image)
                augmented_image = augment_image(image, interference_image, args.d)
                augmented_image.show()
            else:
                print("Must provide interference image to augment the reference image")
        elif args.magnitude_visualize:
            magnitude_image = magnitude_visualize(image)
            magnitude_image.show()
        elif args.magnitude_visualize_raw:
            magnitude_image = magnitude_visualize_raw(image)
            magnitude_image.show()
        elif args.phase_visualize:
            phase_image = phase_visualize(image)
            phase_image.show()
        elif args.magnitude_visualize_rgb:
            magnitude_image = magnitude_visualize_rgb(image)
            magnitude_image.show()
        elif args.high_freq:
            high_freq_img = high_pass_rgb(image, args.d)
            h,w = high_freq_img[0].shape
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(high_freq_img[0])
            rgbArray[:, :, 1] = ifft(high_freq_img[1])
            rgbArray[:, :, 2] = ifft(high_freq_img[2])
            high_freq_img = Image.fromarray(rgbArray)
            high_freq_img.show()
        elif args.low_freq:
            low_freq_img = low_pass_rgb(image, args.d)
            h,w = low_freq_img[0].shape
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(low_freq_img[0])
            rgbArray[:, :, 1] = ifft(low_freq_img[1])
            rgbArray[:, :, 2] = ifft(low_freq_img[2])
            low_freq_img = Image.fromarray(rgbArray)
            low_freq_img.show()
        elif args.perturb_low:
            perturbed_image = perturb_image(image, args.d, perturb_high=False)
            perturbed_image.show()
        elif args.perturb_high:
            perturbed_image = perturb_image(image, args.d, perturb_high=True)
            perturbed_image.show()
        elif args.enhance_high:
            enhanced_image = enhance_high(image, args.d)
            enhanced_image.show()
        elif args.shuffle:
            restrict = False
            shuffleMax = False
            if args.restrict:
                restrict = True
            if args.shuffle_max:
                shuffleMax = True
            shuffled_image = shuffle_image(image, args.d, args.k, args.radius, restrict, shuffleMax)
            shuffled_magnitude_image = magnitude_visualize(shuffled_image)
            shuffled_image.show()
            shuffled_magnitude_image.show()
            # test: what if shuffle high freq image
            '''high_freq_image = high_pass_rgb(image, args.d)
            h,w = high_freq_image[0].shape
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(high_freq_image[0])
            rgbArray[:, :, 1] = ifft(high_freq_image[1])
            rgbArray[:, :, 2] = ifft(high_freq_image[2])
            high_freq_image = Image.fromarray(rgbArray)
            high_freq_magnitude_image = magnitude_visualize(high_freq_image)

            shuffled_image = shuffle_image(high_freq_image, args.d, args.k, args.radius, restrict, shuffleMax)
            shuffled_magnitude_image = magnitude_visualize(shuffled_image)
            fig, axs = plt.subplots(2, 2, figsize=(20,5)) 
            axs[0][0].imshow(high_freq_image, cmap='gray')
            axs[0][0].set_title('High freq img')
            axs[0][0].axis('off')  # Hide the axis

            axs[0][1].imshow(high_freq_magnitude_image, cmap='gray')
            axs[0][1].set_title('High freq img magnitudes')
            axs[0][1].axis('off')

            axs[1][0].imshow(shuffled_image, cmap='gray')
            axs[1][0].set_title('Shuffled image')
            axs[1][0].axis('off')

            axs[1][1].imshow(shuffled_magnitude_image, cmap='gray')
            axs[1][1].set_title('Shuffled image magnitudes')
            axs[1][1].axis('off')

            plt.tight_layout()  # Adjust subplots to fit into the figure area.
            plt.show()  # Display the images'''
        elif args.test:
            # Emphasizing high over low frequency
            '''img_f_shift_rgb = f_shift_rgb(image)
            high_part_r = gaussian_filter_high_pass(img_f_shift_rgb[0].copy(), args.d)*1.5
            high_part_g = gaussian_filter_high_pass(img_f_shift_rgb[1].copy(), args.d)*1.5
            high_part_b = gaussian_filter_high_pass(img_f_shift_rgb[2].copy(), args.d)*1.5
            low_part_r = gaussian_filter_low_pass(img_f_shift_rgb[0].copy(), args.d)*0.5
            low_part_g = gaussian_filter_low_pass(img_f_shift_rgb[1].copy(), args.d)*0.5
            low_part_b = gaussian_filter_low_pass(img_f_shift_rgb[2].copy(), args.d)*0.5
            img_r = high_part_r + low_part_r
            img_g = high_part_g + low_part_g
            img_b = high_part_b + low_part_b
            h,w = img_f_shift_rgb[0].shape
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(img_r)
            rgbArray[:, :, 1] = ifft(img_g)
            rgbArray[:, :, 2] = ifft(img_b)
            test_img = Image.fromarray(rgbArray)
            test_img.show()'''
            # Break down and put image back together in same manner as original paper
            # Verified result image is exactly identical to using above approach
            img_f_shift_rgb = f_shift_rgb(image)
            high_part_r = gaussian_filter_high_pass(img_f_shift_rgb[0].copy(), args.d)
            high_part_g = gaussian_filter_high_pass(img_f_shift_rgb[1].copy(), args.d)
            high_part_b = gaussian_filter_high_pass(img_f_shift_rgb[2].copy(), args.d)
            low_part_r = gaussian_filter_low_pass(img_f_shift_rgb[0].copy(), args.d)
            low_part_g = gaussian_filter_low_pass(img_f_shift_rgb[1].copy(), args.d)
            low_part_b = gaussian_filter_low_pass(img_f_shift_rgb[2].copy(), args.d)
            img_r = high_part_r + low_part_r
            img_g = high_part_g + low_part_g
            img_b = high_part_b + low_part_b
            h,w = img_f_shift_rgb[0].shape
            rgbArray = np.zeros((h,w,3), 'uint8')
            i_img_mix1_fda = ifft(img_r)
            i_img_mix1_fda = np.array(i_img_mix1_fda * 1, np.uint8)
            i_img_mix2_fda = ifft(img_g)
            i_img_mix2_fda = np.array(i_img_mix2_fda * 1, np.uint8)
            i_img_mix3_fda = ifft(img_b)
            i_img_mix3_fda = np.array(i_img_mix3_fda * 1, np.uint8)
            i_img_exp_mix1_fda = np.expand_dims(i_img_mix1_fda, axis=2)
            i_img_exp_mix2_fda = np.expand_dims(i_img_mix2_fda, axis=2)
            i_img_exp_mix3_fda = np.expand_dims(i_img_mix3_fda, axis=2)
            i_img_mix_fda = np.append(i_img_exp_mix1_fda, i_img_exp_mix2_fda, axis=2)
            i_img_mix_fda = np.append(i_img_mix_fda, i_img_exp_mix3_fda, axis=2)
            i_img_mix_fda = Image.fromarray(i_img_mix_fda[:, :, :])
            i_img_mix_fda.show()

    # todo: use refactored functions here
    # maybe one day
    elif args.nuscenes_samples_preprocess:
        samples_dir = '/share/data/nuscenes/samples'
        save_dir = '/share/data/nuscenes/preprocess'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for dir in os.listdir(samples_dir):
            if dir[0:3] == 'CAM': # only process camera images
                print(f'processing {dir}')
                if not os.path.exists(save_dir + '/' + dir):
                    os.makedirs(save_dir + '/' + dir)
                dir_full_path = samples_dir + '/' + dir
                for image_filename in os.listdir(dir_full_path):
                    print(f'processing {image_filename}')
                    pil_image = Image.open(dir_full_path + '/' + image_filename)
                    img_r = np.array(pil_image)[:, :, 0]  # Assuming the red channel is the first channel (index 0), this gets the red channel
                    img_g = np.array(pil_image)[:, :, 1]  # Assuming the green channel is the second channel (index 1), this gets the green channel
                    img_b = np.array(pil_image)[:, :, 2]  # Assuming the blue channel is the third channel (index 2), this gets the blue channel
                    f_r = np.fft.fftn(img_r)
                    f_r_shift = np.fft.fftshift(f_r)
                    f_g = np.fft.fftn(img_g)
                    f_g_shift = np.fft.fftshift(f_g)
                    f_b = np.fft.fftn(img_b)
                    f_b_shift = np.fft.fftshift(f_b)
                    HIGH_D = args.d # d value in high pass filter
                    LOW_D = args.d # d value in low pass filter

                    high_parts_f_r_shift = gaussian_filter_high_pass(f_r_shift.copy(), D=HIGH_D)
                    low_parts_f_r_shift = gaussian_filter_low_pass(f_r_shift.copy(), D=LOW_D)
                    high_parts_f_g_shift = gaussian_filter_high_pass(f_g_shift.copy(), D=HIGH_D)
                    low_parts_f_g_shift = gaussian_filter_low_pass(f_g_shift.copy(), D=LOW_D)
                    high_parts_f_b_shift = gaussian_filter_high_pass(f_b_shift.copy(), D=HIGH_D)
                    low_parts_f_b_shift = gaussian_filter_low_pass(f_b_shift.copy(), D=LOW_D)

                    high_h, high_w = high_parts_f_r_shift.shape # doesn't matter which channel we use here
                    rgbArray = np.zeros((high_h,high_w,3), 'uint8')
                    rgbArray[:, :, 0] = ifft(high_parts_f_r_shift)
                    rgbArray[:, :, 1] = ifft(high_parts_f_g_shift)
                    rgbArray[:, :, 2] = ifft(high_parts_f_b_shift)
                    high_parts_img = Image.fromarray(rgbArray)
                    low_h, low_w = low_parts_f_r_shift.shape # doesn't matter which channel we use here
                    rgbArray = np.zeros((low_h,low_w,3), 'uint8')
                    rgbArray[:, :, 0] = ifft(low_parts_f_r_shift)
                    rgbArray[:, :, 1] = ifft(low_parts_f_g_shift)
                    rgbArray[:, :, 2] = ifft(low_parts_f_b_shift)
                    low_parts_img = Image.fromarray(rgbArray)

                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    high_parts_img.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_highfreq' + image_filename_ext)
                    low_parts_img.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_lowfreq' + image_filename_ext)
    elif args.nuscenes_samples_augment_high:
        data_dir = '/share/code/data/nuscenes/'
        save_dir_name = 'augmented'
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)
        training_filenames = [] # list of all image filenames used for training
        sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']

        if not os.path.exists(data_dir + save_dir_name):
            os.makedirs(data_dir + save_dir_name)
        for sensor in sensors:
            sensor_dir = data_dir + save_dir_name + '/' + sensor
            if not os.path.exists(sensor_dir):
                os.makedirs(sensor_dir)

        for index, scene in enumerate(nusc.scene):
            # go through all samples from training scenes and add their filenames to list
            if scene['name'] in splits.train:
                print(f"{scene['name']} is in splits.train")
                sample = nusc.get('sample', scene['first_sample_token'])
                done = False
                while not done:
                    for sensor in sensors:
                        filename = nusc.get('sample_data', sample['data'][sensor])['filename']
                        training_filenames.append(filename)
                    if sample['next'] == '':
                        done = True
                    else:
                        sample = nusc.get('sample', sample['next'])
            else:
                print(f"{scene['name']} is not in splits.train, skipping")
        print(f'filled training_filenames with {len(training_filenames)} filenames')
        for index, scene in enumerate(nusc.scene):
            if scene['name'] in splits.train:
                # go through training scenes again and generate augmented images
                sample = nusc.get('sample', scene['first_sample_token'])
                interference_filenames = training_filenames.copy()
                done = False
                scene_filenames = []
                while not done:
                    for sensor in sensors:
                        filename = nusc.get('sample_data', sample['data'][sensor])['filename']
                        interference_filenames.remove(filename) # ensure interference candidates are exclusively from different scenes
                        scene_filenames.append(filename)
                    if sample['next'] == '':
                        done = True
                    else:
                        sample = nusc.get('sample', sample['next'])
                
                for filename in scene_filenames:
                    # generate augmented version of each scene image
                    filename_path = pathlib.Path(filename)
                    filename_ext = filename_path.suffix
                    end_of_filename_path = pathlib.Path(*filename_path.parts[1:]) # something like CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883549912464.jpg
                    end_of_filename_path_no_ext = end_of_filename_path.with_suffix('') # get rid of extension
                    save_filename = data_dir + save_dir_name + '/' + str(end_of_filename_path_no_ext) + '_augmented' + str(filename_ext)
                    if not os.path.exists(save_filename):
                        image = Image.open(data_dir + filename) # filename is of the form samples/CAM_BACK/...
                        interference_filename = random.choice(interference_filenames)
                        interference_image = Image.open(data_dir + interference_filename)
                        augmented_image = augment_image(image, interference_image, args.d, augment_high=True)
                        print(f'saving {save_filename}')
                        augmented_image.save(save_filename)
                    else:
                        print(f'{save_filename} already exists, skipping')
    
    elif args.nuscenes_samples_augment_low:
        samples_dir = '/share/data/nuscenes/samples'
        save_dir = '/share/data/nuscenes/augmented_low'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dir in os.listdir(samples_dir):
            if dir[0:3] == 'CAM': # only process camera images
                print(f'processing {dir}')
                if not os.path.exists(save_dir + '/' + dir):
                    os.makedirs(save_dir + '/' + dir)
                dir_full_path = samples_dir + '/' + dir
                filename_list = []
                for image_filename in os.listdir(dir_full_path):
                    filename_list.append(image_filename)
                for image_filename in filename_list:
                    print(f'processing {image_filename}')
                    image = Image.open(dir_full_path + '/' + image_filename)
                    new_filename_list = filename_list.copy()
                    new_filename_list.remove(image_filename) # so don't end up picking same image as interference image
                    interference_image_filename = random.choice(new_filename_list)
                    interference_image = Image.open(dir_full_path + '/' + interference_image_filename)
                    augmented_image = augment_image(image, interference_image, args.d, augment_high=False)
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    augmented_image.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_augmented' + image_filename_ext)
    elif args.nuscenes_samples_perturb_low:
        samples_dir = '/share/data/nuscenes/samples'
        save_dir = '/share/data/nuscenes/perturbed_low'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dir in os.listdir(samples_dir):
            if dir[0:3] == 'CAM': # only process camera images
                print(f'processing {dir}')
                if not os.path.exists(save_dir + '/' + dir):
                    os.makedirs(save_dir + '/' + dir)
                dir_full_path = samples_dir + '/' + dir
                for image_filename in os.listdir(dir_full_path):
                    print(f'processing {image_filename}')
                    image = Image.open(dir_full_path + '/' + image_filename)
                    perturbed_image = perturb_image(image, args.d, perturb_high=False)
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    perturbed_image.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_perturbed' + image_filename_ext)
    elif args.nuscenes_samples_perturb_high:
        samples_dir = '/share/data/nuscenes/samples'
        save_dir = '/share/data/nuscenes/perturbed_high'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dir in os.listdir(samples_dir):
            if dir[0:3] == 'CAM': # only process camera images
                print(f'processing {dir}')
                if not os.path.exists(save_dir + '/' + dir):
                    os.makedirs(save_dir + '/' + dir)
                dir_full_path = samples_dir + '/' + dir
                for image_filename in os.listdir(dir_full_path):
                    print(f'processing {image_filename}')
                    image = Image.open(dir_full_path + '/' + image_filename)
                    perturbed_image = perturb_image(image, args.d, perturb_high=True)
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    perturbed_image.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_perturbed' + image_filename_ext)
    elif args.nuscenes_samples_enhance_high:
        samples_dir = '/share/docker_files/data/nuscenes/samples'
        save_dir = '/share/docker_files/data/nuscenes/enhance_high'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for dir in os.listdir(samples_dir):
            if dir[0:3] == 'CAM': # only process camera images
                print(f'processing {dir}')
                if not os.path.exists(save_dir + '/' + dir):
                    os.makedirs(save_dir + '/' + dir)
                dir_full_path = samples_dir + '/' + dir
                for image_filename in os.listdir(dir_full_path):
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    save_destination = save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_enhanced' + image_filename_ext
                    if not os.path.isfile(save_destination):
                        print(f'processing {image_filename}')
                        image = Image.open(dir_full_path + '/' + image_filename)
                        enhanced_image = enhance_high(image, args.d)
                        enhanced_image.save(save_destination)
                    else:
                        print(f'already processed {image_filename}, skipping')

if __name__ == '__main__':
    main()

