"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from pytorch_fid import fid_score
from torchvision import transforms
import torch
import shutil
import numpy as np
from torchvision.models.inception import inception_v3
from PIL import Image
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def resize_image(image, size=(299, 299)):
    return image.resize(size, Image.BICUBIC)


def check_image_stats(image_dir):
    # Lista delle immagini nella directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Calcola media e deviazione standard delle immagini
    all_images = []
    for image_file in image_files:
        with Image.open(image_file) as img:
            img = img.convert('RGB')  # Assicurati che l'immagine sia in RGB
            img_array = np.array(img) / 255.0  # Normalizza i valori dei pixel
            all_images.append(img_array)
    
    all_images = np.stack(all_images)
    
    mean = np.mean(all_images, axis=(0, 1, 2, 3))
    std = np.std(all_images, axis=(0, 1, 2, 3))
    
    print(f"Mean pixel value: {mean}")
    print(f"Standard deviation of pixel values: {std}")


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Initialize directories for FID calculation
    generated_dir = './generated_images'
    real_dir = './real_images'
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)

    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(model.device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(model.device)


    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))


        # Save generated image for FID calculation
        generated_image = model.fake_B
        real_image = model.real_B

        # Normalize images to [0, 1] range
        generated_image = (generated_image + 1) / 2
        real_image = (real_image + 1) / 2

        # Convert to PIL Image
        generated_image_pil = transforms.ToPILImage()(generated_image.squeeze().cpu())
        real_image_pil = transforms.ToPILImage()(real_image.squeeze().cpu())

        # Resize images to 299x299
        generated_image_resized = resize_image(generated_image_pil)
        real_image_resized = resize_image(real_image_pil)

        # Save resized images
        save_path = os.path.join(generated_dir, f'generated_{i}.png')
        generated_image_resized.save(save_path)

        save_path = os.path.join(real_dir, f'real_{i}.png')
        real_image_resized.save(save_path)

        # Calculate SSIM and PSNR
        ssim_value = ssim(generated_image, real_image).item()
        psnr_value = psnr(generated_image, real_image).item()

        print(f'Image {i}: SSIM = {ssim_value}, PSNR = {psnr_value} dB')

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    print('model')
    print(model.device)
    # Verifica le immagini dopo il ciclo di testing
    print("Verifica delle immagini salvate:")
    check_image_stats(generated_dir)
    check_image_stats(real_dir)

    # Calculate FID after processing all images
    try:
        fid_value = fid_score.calculate_fid_given_paths(paths=(real_dir, generated_dir), batch_size=1, device=model.device, dims=2048)
        print(f'FID: {fid_value}')
    except ValueError as e:
        print(f'Error during FID calculation: {e}')


    # Clean up the generated and real images directories
    shutil.rmtree(generated_dir)
    shutil.rmtree(real_dir)

    webpage.save()  # save the HTML
