import torch
import torch.nn as nn
import config
import os
import random
import numpy as np
import torch.distributed as dist
from collections import OrderedDict
from torchvision.utils import save_image


def save_some_examples(gen, val_loader, epoch, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    image_real, mask = next(iter(val_loader))
    image_real, mask = image_real.to(config.DEVICE), mask.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(mask)
        save_image(denormalize(y_fake), folder + f"/y_gen_{epoch}.png")
        # save_image(CancerInstanceDataset.get_img_mask(mask.cpu()).permute(1, 2, 0).cpu(), folder + f"/input_{epoch}.png")
        if epoch == 0:
            save_image(denormalize(image_real), folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="./model.pth", epoch=-1):
    print("=> Saving model")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, map_location="cpu"):
    print("=> Loading checkpoint", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


# smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(label):
    return label - 0.3 + (torch.rand_like(label) * 0.5)


# smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(label):
    return label + torch.rand_like(label) * 0.3


def denormalize(img):
    """
    :param img: image normalized in [-1, 1]
    :return: img normalized in [0, 1]
    """
    return img * 0.5 + 0.5


def remove_module_key_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key.replace("module.", "")  # removing "module." from key. name = key[7:]
        new_state_dict[name] = value
    return new_state_dict


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", "24129")
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


class RandomRotate90(object):
    """Randomly rotate the input by 90 degrees zero or more times."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return torch.rot90(img, random.randint(0, 3), dims=(1, 2))
        return img