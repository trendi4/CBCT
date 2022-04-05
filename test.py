import argparse
import numpy as np
import time
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


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
parser.add_argument("--dataset_name", type=str, default="pix2pix", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--val_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu to use during training")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--print_interval", type=int, default=1, help="interval between sampling of images from generators")
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
    ImageDataset("data", transforms_ = test_transforms_, mode="val"),
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
        
        input_running_ssim = 0.0
        input_running_psnr = 0.0
        input_running_mse = 0.0
        input_running_nrmse = 0.0

        #DOnt do running values, just add metric values to a list for each batch and then do mean std deviation.

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
                input_img = input[:, 0, :, :].cpu().detach().numpy()
                target_img = target[:, 0, :, :].cpu().detach().numpy()
                pred_img = pred[:, 0, :, :].cpu().detach().numpy()

                #for a in range(opt.batch_size):
                metrics_pred = calculate_metrics(target_img, pred_img)
                metrics_input = calculate_metrics(target_img, input_img)
                
                running_ssim += metrics_pred[0]
                running_psnr += metrics_pred[1]
                running_mse += metrics_pred[2]
                running_nrmse += metrics_pred[3]

                input_running_ssim += metrics_input[0]
                input_running_psnr += metrics_input[1]
                input_running_mse += metrics_input[2]
                input_running_nrmse += metrics_input[3]

                img_sample = torch.cat((input.data, pred.data, target.data, pred.data - input.data, pred.data - target.data), -2)
                save_image(img_sample, "images/%s/%s/val_%s.png" % (opt.dataset_name, "epoch" + split[:-4], i), nrow=5, normalize=True)

                if opt.save_npy:
                    np.save("images/%s/npy/epoch-%s/batch-%d" % (opt.dataset_name, split[:-4], i), pred_img)

            str_log_validate = (
                "Metrics for the prediction: \r[Generator epoch %s] [SSIM %f] [PSNR: %f] [MSE: %f] [NRMSE: %f]"
                % (
                    split[:-4],
                    running_ssim/len(dataloader.dataset),
                    running_psnr/len(dataloader.dataset),
                    running_mse/len(dataloader.dataset),
                    running_nrmse/len(dataloader.dataset)
                )
            )

            str_log_input = (
                "Metrics for the input: \r[Generator epoch %s] [SSIM %f] [PSNR: %f] [MSE: %f] [NRMSE: %f]"
                % (
                    split[:-4],
                    input_running_ssim/len(dataloader.dataset),
                    input_running_psnr/len(dataloader.dataset),
                    input_running_mse/len(dataloader.dataset),
                    input_running_nrmse/len(dataloader.dataset)
                )
            )

            print_log(logger, str_log_input, opt, 1)
            print_log(logger, str_log_validate, opt, 1)
            print_log(logger, "___________________________________________", opt, 1)
            
    logger.close()
