import torch
import wandb
import config
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import os
from dataset.pannuke import PanNuke
from dataset.unitopatho_mask import UnitopathoMasks
from utils import denormalize, remove_module_key_from_state_dict
from generator_model import Generator


def main():
    # wandb_runpath = "daviderubi/unitopatho-generative/3slznzoq"  # scarlet-music-67
    wandb_runpath = "daviderubi/unitopatho-generative/1cvb7cvd"  # smart-lake-59
    path_to_save = "../data/unitopath-public/synthetic_images_smart_lake_59/test"  # path to save images
    top_labels = ["HP", "NORM", "TA.HG", "TA.LG", "TVA.HG", "TVA.LG"]

    # my W&B (Rubinetti)
    wandb.login(key=config.WANDB_KEY_LOGIN)

    num_classes = len(PanNuke.labels())
    gen = Generator(in_channels=num_classes, features=64).to(config.DEVICE)
    load_model(wandb_runpath, "gen.pth", gen)

    # load testset
    test_loader = load_testset()

    # create sub-directories for top labels
    for lbl in top_labels:
        if not os.path.exists(os.path.join(path_to_save, lbl)):
            os.mkdir(os.path.join(path_to_save, lbl))

    # generate synthetic images
    gen.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            mask = sample["mask"].to(config.DEVICE)
            generated_img = gen(mask)

            # save generated images
            for img, image_id, target in zip(generated_img, sample["image_id"], sample["target"]):
                label = top_labels[target.item()]
                save_image(img, os.path.join(path_to_save, label), image_id)

            # log
            if (idx + 1) % 10 == 0:
                print(f"{idx + 1} / {len(test_loader)}")

    wandb.finish()


def save_image(img, path, image_id):
    name = image_id.split('/')[-1].strip("'")
    file_name = os.path.join(path, name)
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
    test_dataset = UnitopathoMasks(df, T=transform_test, path=path, target='top_label', path_masks=path_masks,
                                   train=False, device=torch.cuda.current_device())
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)
    return test_loader


def load_model(run_path, file_name, model):
    print("=> Loading", file_name)
    # download model from wandb
    api = wandb.Api()
    run = api.run(run_path)
    run.file(file_name).download(replace=True)

    checkpoint = torch.load(file_name)
    state_dict = checkpoint["state_dict"]

    # When using DistributedDataParallel, it is proper to save the model by using:
    # torch.save(model.module.state_dict(), 'file_name.pt'), instead of:
    # torch.save(model.state_dict(), 'file_name.pt').
    # Unfortunately we saved the model in the second way, so we have to remove
    # "module." from the keys of state_dict.
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    new_state_dict = remove_module_key_from_state_dict(state_dict)

    model.load_state_dict(new_state_dict)


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    main()
