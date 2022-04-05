import argparse
import numpy as np
import time
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse


from model import *
from dataset import *
from util import *



assert(torch.cuda.is_available()), "No GPU available"
device = "cuda" if torch.cuda.is_available() else "cpu"

#######################
### Parse arguments ###
#######################


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="pix2pix", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--val_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--lambda_pixel", type=int, default=100, help="weight of the L1-loss component")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu to use during training")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--print_interval", type=int, default=1, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--save_npy", type=bool, default=False, help="save numpy files to folder")

opt = parser.parse_args()
print(opt)

# Initialize model 
generator = GeneratorUNet()
discriminator = Discriminator()

# Initalize Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Move everything to GPU for faster training
if device == "cuda":
    generator = generator.to(device)

# If multiple GPUs available, we make sure to parallelize the data
if opt.n_gpu > 1:
    generator = nn.DataParallel(generator)


test_transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.CenterCrop(256)
]


dataloader = DataLoader(
    ImageDataset("data", transforms_ = test_transforms_, mode="train"),
    batch_size=opt.val_batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

if opt.save_npy:
    create_folder("images/%s/npy" % (opt.dataset_name))




if __name__ == "__main__":

    gen_list = [f for f in glob.glob("saved_models/%s/generator_*.pth" % (opt.dataset_name))]

    logger = open(os.path.join("data", 'validation_log.txt'), 'w+')
    print_log(logger, opt, opt)

    for it, file in enumerate(gen_list):

        generator.load_state_dict(torch.load(file))

        running_ssim = 0.0
        running_psnr = 0.0
        running_mse = 0.0
        running_nrmse = 0.0
        split = file.split('_')[-1]
        create_folder("images/%s/epoch%s" % (opt.dataset_name, split[:-4]))

        if opt.save_npy:
            create_folder("images/%s/npy/epoch-%s" % (opt.dataset_name, split[:-4]))


        print_log(logger, "Using the generator and discriminator from the {} epoch".format(split[:-4]), opt, 1)

        with torch.no_grad():
            generator.eval()

            for i, batch in enumerate(dataloader):
                # Model inputs
                input = batch[0].to(device)
                target = batch[1].to(device)

                pred = generator(input)

                target_img = target[:, 0, :, :].cpu().detach().numpy()
                pred_img = pred[:, 0, :, :].cpu().detach().numpy()

                #for a in range(opt.batch_size):
                a = 0
                range = (pred_img[a, :, :]).max() - (pred_img[a, :, :]).min()
                ssim_val = ssim(target_img[a, :, :], pred_img[a, :, :], data_range= range)
                psnr_val = psnr(target_img[a, :, :], pred_img[a, :, :], data_range= range)
                mse_val = mse(target_img[a, :, :], pred_img[a, :, :])
                nrmse_val = nrmse(target_img[a, :, :], pred_img[a, :, :])
                
                running_ssim += ssim_val
                running_psnr += psnr_val
                running_mse += mse_val
                running_nrmse += nrmse_val

                img_sample = torch.cat((input.data, pred.data, target.data, pred.data - input.data, pred.data - target.data), -2)
                save_image(img_sample, "images/%s/%s/val_%s.png" % (opt.dataset_name, "epoch" + split[:-4], i), nrow=5, normalize=True)

                if opt.save_npy:
                    np.save("images/%s/npy/epoch-%s/batch-%d" % (opt.dataset_name, split[:-4], i), pred_img)

            str_log = (
                "\r[Generator epoch %s] [SSIM %f] [PSNR: %f] [MSE: %f] [NRMSE: %f]"
                % (
                    split[:-4],
                    running_ssim/len(dataloader.dataset),
                    running_psnr/len(dataloader.dataset),
                    running_mse/len(dataloader.dataset),
                    running_nrmse/len(dataloader.dataset)
                )
            )
            print_log(logger, str_log, opt, 1)
            
    logger.close()
