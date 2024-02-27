from __future__ import print_function
import argparse
import numpy as np
import torch
import random
import os
import pathlib
import random
from PIL import Image

# get the low frequency using Gaussian Low-Pass Filter
# high d value tends to make high frequency info empty, low frequency info less blurred (intensifies high filter, weakens low filter)
def gaussian_filter_low_pass(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to dis_square
    return template * fshift

# get the high frequency using Gaussian High-Pass Filter
def gaussian_filter_high_pass(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to 0
    return template * fshift

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

def freq_domain(img):
    '''Convert image to grayscale and return frequency domain representation of image.'''
    image_grayscale = img.convert('L')
    img_arr = np.array(image_grayscale)
    img_arr_f_shift = f_shift(img_arr)
    # Calculate the magnitude spectrum, scaled so that it is useful for visualization
    magnitude_spectrum = 14*np.log(np.abs(img_arr_f_shift))
    freq_image = Image.fromarray(magnitude_spectrum)
    return freq_image

def freq_domain_rgb(img):
    img_f_shift_rgb = f_shift_rgb(img)
    h, w = img_f_shift_rgb[0].shape # doesn't matter which channel we use here
    rgbArray = np.zeros((h,w,3), 'uint8')
    rgbArray[:, :, 0] = 14*np.log(np.abs(img_f_shift_rgb[0]))
    rgbArray[:, :, 1] = 14*np.log(np.abs(img_f_shift_rgb[1]))
    rgbArray[:, :, 2] = 14*np.log(np.abs(img_f_shift_rgb[2]))
    freq_image = Image.fromarray(rgbArray)
    return freq_image

def low_pass_rgb(img, D):
    img_f_shift_rgb = f_shift_rgb(img)
    low_part_r = gaussian_filter_low_pass(img_f_shift_rgb[0].copy(), D=D)
    low_part_g = gaussian_filter_low_pass(img_f_shift_rgb[1].copy(), D=D)
    low_part_b = gaussian_filter_low_pass(img_f_shift_rgb[2].copy(), D=D)
    low_rgb = np.array([low_part_r, low_part_g, low_part_b])
    return low_rgb

def high_pass_rgb(img, D):
    img_f_shift_rgb = f_shift_rgb(img)
    high_part_r = gaussian_filter_high_pass(img_f_shift_rgb[0].copy(), D=D)
    high_part_g = gaussian_filter_high_pass(img_f_shift_rgb[1].copy(), D=D)
    high_part_b = gaussian_filter_high_pass(img_f_shift_rgb[2].copy(), D=D)
    high_rgb = np.array([high_part_r, high_part_g, high_part_b])
    return high_rgb

def low_high_pass_rgb(img, D):
    low_rgb = low_pass_rgb(img, D)
    high_rgb = high_pass_rgb(img, D)
    low_high_rgb = np.array([low_rgb, high_rgb])
    return low_high_rgb

def augment_image(r_img, i_img, D): # reference image, interference image
    channel_num = random.randrange(0,3) # Randomize which channel to interfere with - 0 is red, 1 is green, 2 is blue
    i_img_low_high_parts = low_high_pass_rgb(i_img, D)
    r_img_low_high_parts = low_high_pass_rgb(r_img, D)

    augmented_low_high_parts_r = r_img_low_high_parts[0][0] + (i_img_low_high_parts[1][0] if channel_num == 0 else r_img_low_high_parts[1][0])
    augmented_low_high_parts_g = r_img_low_high_parts[0][1] + (i_img_low_high_parts[1][1] if channel_num == 1 else r_img_low_high_parts[1][1])
    augmented_low_high_parts_b = r_img_low_high_parts[0][2] + (i_img_low_high_parts[1][2] if channel_num == 2 else r_img_low_high_parts[1][2])
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

def augment_image_low(r_img, i_img, D): # reference image, interference image
    channel_num = random.randrange(0,3) # Randomize which channel to interfere with - 0 is red, 1 is green, 2 is blue
    i_img_low_high_parts = low_high_pass_rgb(i_img, D)
    r_img_low_high_parts = low_high_pass_rgb(r_img, D)

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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FFT/iFFT test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--reference-image', type=str, metavar='filename',
                        help='reference image to transform or augment with interference image')
    parser.add_argument('--interference-image', type=str, metavar='filename',
                        help='interference image used to augment reference image')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--d', type=int, default=10, metavar='S',
                        help='d value in gaussian function')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #parser.add_argument('--no-mps', action='store_true', default=False,
    #                    help='disables macOS GPU training')
    parser.add_argument('--show-transforms', action='store_true', default=False,
                        help='show high and low pass transforms on each channel of image')
    parser.add_argument('--augment-image', action='store_true', default=False,
                        help='show image augmented in the same manner as the FDA paper')
    parser.add_argument('--freq-domain', action='store_true', default=False,
                        help='show frequency domain representation of grayscaled image')
    parser.add_argument('--freq-domain-rgb', action='store_true', default=False,
                        help='show frequency domain representation of image (RGB)')
    parser.add_argument('--high-freq', action='store_true', default=False,
                        help='get the high frequency components of the reference image in RGB')
    parser.add_argument('--low-freq', action='store_true', default=False,
                        help='get the low frequency components of the reference image in RGB')
    parser.add_argument('--nuscenes-samples-preprocess', action='store_true', default=False,
                        help='save low and high frequency components of nuscenes samples to new directory "preproces"')
    parser.add_argument('--nuscenes-samples-augment-high', action='store_true', default=False,
                        help='save randomly high-frequency augmented versions of images')
    parser.add_argument('--nuscenes-samples-augment-low', action='store_true', default=False,
                        help='save randomly low-frequency augmented versions of images')
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
        elif args.freq_domain:
            freq_image = freq_domain(image)
            freq_image.show()
        elif args.freq_domain_rgb:
            freq_image_rgb = freq_domain_rgb(image)
            freq_image_rgb.show()
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
    # todo: use refactored functions here
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
        samples_dir = '/share/data/nuscenes/samples'
        save_dir = '/share/data/nuscenes/augmented_high'
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
                    new_filename_list = filename_list
                    new_filename_list.remove(image_filename) # so don't end up picking same image as interference image
                    interference_image_filename = random.choice(new_filename_list)
                    interference_image = Image.open(dir_full_path + '/' + interference_image_filename)
                    augmented_image = augment_image(image, interference_image, args.d)
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    augmented_image.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_augmented' + image_filename_ext)
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
                    new_filename_list = filename_list
                    new_filename_list.remove(image_filename) # so don't end up picking same image as interference image
                    interference_image_filename = random.choice(new_filename_list)
                    interference_image = Image.open(dir_full_path + '/' + interference_image_filename)
                    augmented_image = augment_image_low(image, interference_image, args.d)
                    image_filename_path = pathlib.Path(image_filename)
                    image_filename_no_ext = image_filename_path.with_suffix('') # get rid of extension
                    image_filename_ext = image_filename_path.suffix
                    augmented_image.save(save_dir + '/' + dir + '/' + str(image_filename_no_ext) + '_augmented' + image_filename_ext)

if __name__ == '__main__':
    main()

