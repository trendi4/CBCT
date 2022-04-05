import os 
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
import shutil

def create_initial_directories(opt):
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("images/%s/loss_plots" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
    
def sample_images(batches_done, dataloader, generator, opt):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(dataloader))
    input = imgs[0].to("cuda")
    target = imgs[1].to("cuda")
    pred = generator(input)
    img_sample = torch.cat((input.data, pred.data, target.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

def print_log(logger, message, opt, batches_done = -1):
    if (batches_done %  opt.print_interval) == 0 and batches_done != -1:
        print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

def draw_loss(G_loss, D_loss, epoch, name):
    plt.plot(G_loss, '-b', label='G_loss')
    plt.plot(D_loss, '-r', label='D_loss')

    plt.xlabel("Epoch")
    plt.title("Loss plot")

    if not epoch:
        plt.legend(loc='upper right')

    # save image
    plt.savefig(name + ".png")

def create_folder(name): 
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)