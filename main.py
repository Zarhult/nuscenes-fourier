from __future__ import print_function
import argparse
import numpy as np
import torch
import sys
import random
import os
import pathlib
from PIL import Image
from matplotlib import pyplot as plt

print_stuff = False # print image arrays in filter process, use with small images because it prints every corresponding pixel value

if print_stuff:
    np.set_printoptions(threshold=sys.maxsize)

# get the low frequency using Gaussian Low-Pass Filter
def gaussian_filter_low_pass(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to dis_square
    if print_stuff:
        print("LOW PASS:")
        print(f"center: \n{center}")
        print(f"dis_square: \n{dis_square}")
        print(f"template: \n{template}")

    return template * fshift

# get the high frequency using Gaussian High-Pass Filter
def gaussian_filter_high_pass(fshift, D):
    h, w = fshift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(- dis_square / (2 * D ** 2)) # larger D value will make template closer to 0
    if print_stuff:
        print("HIGH PASS:")
        print(f"center: \n{center}")
        print(f"dis_square: \n{dis_square}")
        print(f"template: \n{template}")

    return template * fshift

# Inverse Fourier transform
def ifft(fshift):
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifftn(ishift)
    iimg = np.abs(iimg)

    return iimg

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
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
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
        img_r = np.array(image)[:, :, 0]  # Assuming the red channel is the first channel (index 0), this gets the red channel
        img_g = np.array(image)[:, :, 1]  # Assuming the green channel is the second channel (index 1), this gets the green channel
        img_b = np.array(image)[:, :, 2]  # Assuming the blue channel is the third channel (index 2), this gets the blue channel
        f_r = np.fft.fftn(img_r)
        f_r_shift = np.fft.fftshift(f_r)
        f_g = np.fft.fftn(img_g)
        f_g_shift = np.fft.fftshift(f_g)
        f_b = np.fft.fftn(img_b)
        f_b_shift = np.fft.fftshift(f_b)
        # high d value tends to make high frequency info empty, low frequency info less blurred (intensifies high filter, weakens low filter)
        # and vice versa
        HIGH_D = args.d # d value in high pass filter
        LOW_D = args.d # d value in low pass filter

        high_parts_f_r_shift = gaussian_filter_high_pass(f_r_shift.copy(), D=HIGH_D)
        low_parts_f_r_shift = gaussian_filter_low_pass(f_r_shift.copy(), D=LOW_D)
        high_parts_f_g_shift = gaussian_filter_high_pass(f_g_shift.copy(), D=HIGH_D)
        low_parts_f_g_shift = gaussian_filter_low_pass(f_g_shift.copy(), D=LOW_D)
        high_parts_f_b_shift = gaussian_filter_high_pass(f_b_shift.copy(), D=HIGH_D)
        low_parts_f_b_shift = gaussian_filter_low_pass(f_b_shift.copy(), D=LOW_D)

        if args.show_transforms:
            # Create images from each transform
            high_r_img = Image.fromarray(ifft(high_parts_f_r_shift))
            low_r_img = Image.fromarray(ifft(low_parts_f_r_shift))
            high_g_img = Image.fromarray(ifft(high_parts_f_g_shift))
            low_g_img = Image.fromarray(ifft(low_parts_f_g_shift))
            high_b_img = Image.fromarray(ifft(high_parts_f_b_shift))
            low_b_img = Image.fromarray(ifft(low_parts_f_b_shift))

            # Display original and new images side by side, each channel
            fig, axes = plt.subplots(3, 3, figsize=(20, 10))

            axes[0, 0].set_title('Original image')
            # Original images
            axes[0, 0].set_ylabel('Red channel')
            axes[0, 0].imshow(img_r)
            axes[1, 0].set_ylabel('Green channel')
            axes[1, 0].imshow(img_g)
            axes[2, 0].set_ylabel('Blue channel')
            axes[2, 0].imshow(img_b)

            axes[0, 1].set_title('High-pass transformed')
            # High-pass transformed images
            axes[0, 1].imshow(high_r_img)
            axes[1, 1].imshow(high_g_img)
            axes[2, 1].imshow(high_b_img)

            axes[0, 2].set_title('Low-pass transformed')
            # Low-pass transformed images
            axes[0, 2].imshow(low_r_img)
            axes[1, 2].imshow(low_g_img)
            axes[2, 2].imshow(low_b_img)

            plt.show()
        elif args.augment_image:
            # Add high and low pass transforms for each channel together, and splice in one channel from an interference image.
            # Then iFFT to convert back to regular image.

            if args.interference_image is not None:
                interference_image = Image.open(args.interference_image)
                channel_num = random.randrange(0,3) # Randomize which channel to interfere with - 0 is red, 1 is green, 2 is blue
                interference_image_r = np.array(interference_image)[:, :, 0]  # Assuming the red channel is the first channel (index 0)
                interference_image_g = np.array(interference_image)[:, :, 1]  # Assuming the green channel is the second channel (index 1)
                interference_image_b = np.array(interference_image)[:, :, 2]  # Assuming the blue channel is the third channel (index 2)
                interference_f_r = np.fft.fftn(interference_image_r)
                interference_f_r_shift = np.fft.fftshift(interference_f_r)
                interference_high_parts_f_r_shift = gaussian_filter_high_pass(interference_f_r_shift.copy(), D=HIGH_D)
                interference_f_g = np.fft.fftn(interference_image_g)
                interference_f_g_shift = np.fft.fftshift(interference_f_g)
                interference_high_parts_f_g_shift = gaussian_filter_high_pass(interference_f_g_shift.copy(), D=HIGH_D)
                interference_f_b = np.fft.fftn(interference_image_b)
                interference_f_b_shift = np.fft.fftshift(interference_f_b)
                interference_high_parts_f_b_shift = gaussian_filter_high_pass(interference_f_b_shift.copy(), D=HIGH_D)

                low_high_f_r_shift = low_parts_f_r_shift + (interference_high_parts_f_r_shift if channel_num == 0 else high_parts_f_r_shift)
                low_high_f_g_shift = low_parts_f_g_shift + (interference_high_parts_f_g_shift if channel_num == 1 else high_parts_f_g_shift)
                low_high_f_b_shift = low_parts_f_b_shift + (interference_high_parts_f_b_shift if channel_num == 2 else high_parts_f_b_shift)
                img_r = ifft(low_high_f_r_shift)
                img_g = ifft(low_high_f_g_shift)
                img_b = ifft(low_high_f_b_shift)
                h, w = low_high_f_r_shift.shape # doesn't matter which channel we use here
                rgbArray = np.zeros((h,w,3), 'uint8')
                rgbArray[:, :, 0] = img_r
                rgbArray[:, :, 1] = img_g
                rgbArray[:, :, 2] = img_b
                img = Image.fromarray(rgbArray)
                img.show()
            else:
                print("Must provide interference image to augment the reference image")
        elif args.freq_domain:
            image_grayscale = image.convert('L')
            img_arr = np.array(image_grayscale)
            img_arr_fft = np.fft.fftn(img_arr)
            img_arr_fft_shift = np.fft.fftshift(img_arr_fft)
            # Calculate the magnitude spectrum, scaled so that it is useful for visualization
            magnitude_spectrum = 14*np.log(np.abs(img_arr_fft_shift))
            f_r_shift_img = Image.fromarray(magnitude_spectrum)
            f_r_shift_img.show()
        elif args.freq_domain_rgb:
            h, w = f_r_shift.shape # doesn't matter which channel we use here
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = 14*np.log(np.abs(f_r_shift))
            rgbArray[:, :, 1] = 14*np.log(np.abs(f_g_shift))
            rgbArray[:, :, 2] = 14*np.log(np.abs(f_b_shift))
            img = Image.fromarray(rgbArray)
            img.show()
        elif args.high_freq:
            h, w = high_parts_f_r_shift.shape # doesn't matter which channel we use here
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(high_parts_f_r_shift)
            rgbArray[:, :, 1] = ifft(high_parts_f_g_shift)
            rgbArray[:, :, 2] = ifft(high_parts_f_b_shift)
            img = Image.fromarray(rgbArray)
            img.show()
        elif args.low_freq:
            h, w = low_parts_f_r_shift.shape # doesn't matter which channel we use here
            rgbArray = np.zeros((h,w,3), 'uint8')
            rgbArray[:, :, 0] = ifft(low_parts_f_r_shift)
            rgbArray[:, :, 1] = ifft(low_parts_f_g_shift)
            rgbArray[:, :, 2] = ifft(low_parts_f_b_shift)
            img = Image.fromarray(rgbArray)
            img.show()

    elif args.nuscenes_samples_preprocess:
        samples_dir = './samples'
        save_dir = './preprocess'
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
                    HIGH_D = 10
                    LOW_D = 10

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

if __name__ == '__main__':
    main()
