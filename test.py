import torch
import wandb
import config
from dataset.pannuke import CancerInstanceDataset
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import os
from dataset.unitopatho_mask import UTP_Masks
from utils import denormalize
from collections import OrderedDict
from generator_model import Generator
from PIL import Image
import time

def main():
    # my W&B (Rubinetti)
    wandb.login(key="58214c04801c157c99c68d2982affc49dd6e4072")

    num_classes = len(CancerInstanceDataset.labels())
    gen = Generator(in_channels=num_classes, features=64).to(config.DEVICE)
    load_model("gen.pth", gen)

    # load testset
    test_loader = load_testset()

    # generate synthetic images
    gen.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            mask = sample["mask"].to(config.DEVICE)
            generated_img = gen(mask)

            # save generated images
            for img, image_id in zip(generated_img, sample["image_id"]):
                save_image(img, image_id)

            # log
            if (idx + 1) % 10 == 0:
                print(f"{idx + 1} / {len(test_loader)}")

    wandb.finish()


def save_image(img, image_id):
    file_name = image_id.split('/')[-1].strip("'")
    file_name = f'../data/unitopath-public/synthetic_images/test/{file_name}'
    img = transforms.CenterCrop(1812)(img)  # bring the image to the original size (1812x1812)
    img = denormalize(img)  # normalize to [0, 1]
    torchvision.utils.save_image(img, file_name)


def load_testset():
    path = '../data/unitopath-public/800'
    path_masks = "../data/unitopath-public/generated_torchstain"
    padding = int((2048 - 1812) / 2)
    transform_test = transforms.Compose([
        transforms.Pad(padding, padding_mode="reflect"),  # we add padding to bring the images to 2048x2048
    ])
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    df = df[df.grade >= 0].copy()
    test_dataset = UTP_Masks(df, T=transform_test, path=path, target='grade', path_masks=path_masks, train=False,
                             device=torch.cuda.current_device())
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)
    return test_loader


def load_model(file_name, model):
    print("=> Loading", file_name)
    # download model from wandb
    api = wandb.Api()
    run = api.run("daviderubi/unitopatho-generative/3slznzoq")  # scarlet-music-67
    run.file(file_name).download(replace=True)

    checkpoint = torch.load(file_name)
    state_dict = checkpoint["state_dict"]

    # When using DistributedDataParallel, it is proper to save the model by using:
    # torch.save(model.module.state_dict(), 'file_name.pt'), instead of:
    # torch.save(model.state_dict(), 'file_name.pt').
    # Unfortunately we saved the model in the second way, so we have to remove
    # "module." from the keys of state_dict.
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key.replace("module.", "")  # removing "module." from key. name = key[7:]
        new_state_dict[name] = value

    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    main()