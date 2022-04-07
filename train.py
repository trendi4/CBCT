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
from util import create_initial_directories, draw_loss, draw_metrics, sample_images, print_log, calculate_metrics



assert(torch.cuda.is_available()), "No GPU available"
device = "cuda" if torch.cuda.is_available() else "cpu"

#######################
### Parse arguments ###
#######################


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="pix2pix", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--val_batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--lambda_pixel", type=int, default=100, help="weight of the L1-loss component")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=4, help="number of gpu to use during training")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--print_interval", type=int, default=1, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--validate_interval", type=int, default=1, help="interval between model validations")

opt = parser.parse_args()
print(opt)

# Create images/ and saved_models/ to store validation samples and weight values
create_initial_directories(opt)

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
    discriminator = discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)

# If multiple GPUs available, we make sure to parallelize the data
if opt.n_gpu > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    criterion_GAN = nn.DataParallel(criterion_GAN)
    criterion_pixelwise = nn.DataParallel(criterion_pixelwise)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.decay_epoch, gamma=0.5)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=opt.decay_epoch, gamma=0.5)
train_transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(30)
]

val_transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.CenterCrop(256)
]


val_dataloader = DataLoader(
    ImageDataset("data", transforms_= train_transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

dataloader = DataLoader(
    ImageDataset("data", transforms_ = val_transforms_, mode="val"),
    batch_size=opt.val_batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)



if __name__ == "__main__":

    prev_time = time.time()
    load_start_time = time.time()

    values_loss_G = []
    values_loss_D = []


    logger = open(os.path.join("data", 'log.txt'), 'w+')
    print_log(logger, opt, opt)
    

    for epoch in range(opt.epoch, opt.n_epochs):

        running_loss_G = 0.0
        running_loss_D = 0.0
 
        for i, batch in enumerate(dataloader):
            load_end_time = time.time()
            # Model inputs
            input = batch[0].to(device)
            target = batch[1].to(device)

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((input.size(0), *patch))).to(device)
            fake = torch.Tensor(np.zeros((input.size(0), *patch))).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            pred = generator(input)
            pred_fake = discriminator(pred, input)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(pred, target)

            # Total loss
            loss_G = loss_GAN + opt.lambda_pixel * loss_pixel
            running_loss_G += loss_G.item() * pred.size(0)

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(target, input)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(pred.detach(), input)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            running_loss_D += loss_D.item() * pred.size(0)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            loading_time = load_start_time - load_end_time

            # Print log
            str_log = (
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s batch load time: %.4f batch run time: %.4f"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item() * opt.lambda_pixel,
                    loss_GAN.item(),
                    time_left,
                    -loading_time,
                    prev_time - load_end_time
                )
            )

            print_log(logger, str_log, opt, batches_done)

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done, val_dataloader, generator, opt)
            
            load_start_time=time.time()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch+1))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch+1))
        
        if epoch % opt.validate_interval == 0:
            generator.eval()

            running_ssim = []
            running_psnr = []
            running_mse = []
            running_nrmse = []    

            with torch.no_grad():
                for ii, val_batch in enumerate(val_dataloader):
                    input = val_batch[0].to(device)
                    target = val_batch[1].to(device)

                    name = batch[2][0].split('\\')[-1]
                    pred = generator(input)
                    target_img = target[:, 0, :, :].cpu().detach().numpy()
                    pred_img = pred[:, 0, :, :].cpu().detach().numpy()

                    #for a in range(opt.batch_size):
                    metrics_pred = calculate_metrics(target_img[0], pred_img[0])
                    
                    running_ssim.append(metrics_pred[0])
                    running_psnr.append(metrics_pred[1])
                    running_mse.append(metrics_pred[2])
                    running_nrmse.append(metrics_pred[3])

                str_log_validate = (
                    "Validation mettrics for epoch %s: [SSIM %f +/- %f] [PSNR: %f +/- %f] [MSE: %f +/- %f] [NRMSE: %f +/- %f]"
                    % (
                        epoch+1,
                        sum(running_ssim)/len(dataloader.dataset), np.std(running_ssim),
                        sum(running_psnr)/len(dataloader.dataset), np.std(running_psnr),
                        sum(running_mse)/len(dataloader.dataset), np.std(running_mse),
                        sum(running_nrmse)/len(dataloader.dataset), np.std(running_nrmse)
                    )
                )

                print_log(logger, "___________________________________________", opt, 1)
                print_log(logger, str_log_validate, opt, 1)
                print_log(logger, "___________________________________________", opt, 1)

                draw_metrics(sum(running_ssim)/len(dataloader.dataset), sum(running_psnr)/len(dataloader.dataset), sum(running_mse)/len(dataloader.dataset), sum(running_nrmse)/len(dataloader.dataset), epoch, "images/%s/loss_plots/metrics_epoch-%d" % (opt.dataset_name, epoch+1) )

        scheduler1.step()
        scheduler2.step()

        values_loss_D.append(running_loss_D)
        values_loss_G.append(running_loss_G)
        draw_loss(values_loss_G, values_loss_D, epoch, "images/%s/loss_plots/loss_epoch-%d" % (opt.dataset_name, epoch+1))

    logger.close()