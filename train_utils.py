import os
import torchvision
import config
import wandb
import torch
import utils
import numpy as np
import pandas as pd
from dataset.pannuke import PanNuke
from dataset.unitopatho_mask import UnitopathoMasks
from torchvision.transforms import transforms
from tqdm import tqdm


# Initialize wandb project. Use host="https://wandb.opendeephealth.di.unito.it" to use wandb on open deep health.
def wandb_init(wandb_key_login, project_name, host=None):
    wandb.login(host=host, key=wandb_key_login)
    wandb.init(project=project_name,
               config={
                   "seed": config.SEED,
                   "device": config.DEVICE,
                   "root": config.ROOT_DIR,
                   "epochs": config.NUM_EPOCHS,
                   "lr": config.LEARNING_RATE,
                   "num_workers": config.NUM_WORKERS,
                   "l1_lambda": config.L1_LAMBDA,
                   "adam_beta1": config.ADAM_BETA1,
                   "adam_beta2": config.ADAM_BETA2,
                   "batch_size": config.BATCH_SIZE,
                   "he_norm": config.HE_NORM,
                   "load_model": config.LOAD_MODEL,
                   "smooth_positive_labels": config.SMOOTH_POSITIVE_LABELS,
                   "smooth_negative_labels": config.SMOOTH_NEGATIVE_LABELS})
    config.CHECKPOINT_GEN = os.path.join(wandb.run.dir, config.CHECKPOINT_GEN)
    config.CHECKPOINT_DISC = os.path.join(wandb.run.dir, config.CHECKPOINT_DISC)


def wandb_get_images_to_log(loader, num_img=10):
    """
    :param loader: loader of the dataset.
    :param num_img: how many images you want to log.
    :return: a batch containing num_img images and masks thought to be logged.
    """
    imgs = []
    masks = []
    count = 0

    for sample in loader:
        imgs.append(sample["image"])
        masks.append(sample["mask"])
        count += sample["image"].size()[0]
        if count >= num_img:
            break

    test_batch_im = torch.cat(imgs, dim=0)[:num_img]
    test_batch_mask = torch.cat(masks, dim=0)[:num_img]
    test_batch_im = utils.denormalize(test_batch_im)
    return test_batch_im.cpu(), test_batch_mask.cpu()


def wandb_load_model(run_path, file_name, model, optimizer, lr, map_location, remove_module_key=False):
    """
    It loads a model from wandb.
    :param run_path: The wandb project run path from which take the model.
    :param file_name: The filename used to save the model on wandb.
    :param model: The model to load the state_dict.
    :param optimizer: The optimizer to load the state_dict.
    :param lr: Learning rate value to be set to the optimizer.
    :param map_location: Map Location.
    :param remove_module_key: True if you want to remove "module." key state_dict (see details in the code).
    """
    print("=> Loading ", file_name)
    api = wandb.Api()
    run = api.run(run_path)  # upbeat-river-42
    run.file(file_name).download(replace=True)

    checkpoint = torch.load(file_name, map_location=map_location)
    state_dict = checkpoint["state_dict"]

    if remove_module_key:
        # When using DistributedDataParallel, it is proper to save the model by using:
        # torch.save(model.module.state_dict(), 'file_name.pt'), instead of:
        # torch.save(model.state_dict(), 'file_name.pt').
        # Unfortunately we saved the model in the second way, so we have to remove
        # "module." from the keys of state_dict.
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
        state_dict = utils.remove_module_key_from_state_dict(state_dict)

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Log on wandb the losses and the generated images for test_batch_masks.
def wandb_log_epoch(gen, test_batch_masks, g_adv_loss, g_l1_loss, d_loss):
    gen.eval()
    with torch.no_grad():
        fakes = gen(test_batch_masks.to(config.DEVICE))
        fakes = utils.denormalize(fakes)
        wandb.log({"generator_adv_loss": g_adv_loss,
                   "generator_l1_loss": g_l1_loss,
                   "discriminator_loss": d_loss,
                   "Fakes": wandb.Image(torchvision.utils.make_grid(fakes))})
    gen.train()


# Log some generated images on wandb.
def wandb_log_generated_images(gen, loader, batch_to_log=5):
    images_to_log = []

    gen.eval()
    with torch.no_grad():
        for idx, sample in enumerate(loader):
            reals = sample["image"].to(config.DEVICE)
            masks = sample["mask"].to(config.DEVICE)
            fakes = gen(masks)

            for fake, real, mask in zip(fakes, reals, masks):  # for each element in batch
                mask = PanNuke.get_img_mask(mask.cpu()).cpu()
                real = utils.denormalize(real).cpu()
                fake = utils.denormalize(fake).cpu()
                images_to_log.append(torchvision.utils.make_grid([mask, real, fake]))

            if idx + 1 == batch_to_log:
                break

    wandb.log({"Generated_images (mask-real-fake)": [wandb.Image(img, caption="Mask - Real - Fake")
                                                     for img in images_to_log]})
    gen.train()


# Train generator and discriminator for an epoch.
def train_epoch(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, gpu):
    loop = tqdm(loader, leave=True)
    do_log = torch.cuda.current_device() == 0
    disc_losses = []
    gen_l1_losses = []
    gen_adv_losses = []
    gen.train()
    disc.train()
    # if FiveCrop is used is transformations, we need to fuse batch_size and ncrops dimensions in the loop
    five_crop = any(isinstance(tr, transforms.FiveCrop) for tr in loader.dataset.transform.transforms)

    for idx, sample in enumerate(loop):
        real_image = sample["image"].cuda(gpu)
        mask = sample["mask"].cuda(gpu)

        # fuse batch size and ncrops
        if five_crop:
            bs, ncrops, c_img, h, w = real_image.size()
            c_mask = mask.size()[2]
            real_image = real_image.view(-1, c_img, h, w)  # bs * ncrops, c, h, w
            mask = mask.view(-1, c_mask, h, w)  # bs * ncrops, c, h, w

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake_image = gen(mask)
            # real batch
            d_real = disc(mask, real_image)
            target = torch.ones_like(d_real)
            if config.SMOOTH_POSITIVE_LABELS:
                target = utils.smooth_positive_labels(target)
            d_real_loss = bce(d_real, target)
            # fake batch
            d_fake = disc(mask, fake_image.detach())
            target = torch.zeros_like(d_fake)
            if config.SMOOTH_NEGATIVE_LABELS:
                target = utils.smooth_negative_labels(target)
            d_fake_loss = bce(d_fake, target)
            # the Pix2pix GAN paper says: "we divide the objective by 2 while optimizing D, which slows down the rate at
            # which D learns relative to G"
            d_loss = (d_real_loss + d_fake_loss) / 2
            disc_losses.append(d_loss.item())

        # Discriminator weights update.
        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = disc(mask, fake_image)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(fake_image, real_image) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1
            gen_l1_losses.append(l1.item())
            gen_adv_losses.append(g_fake_loss.item())

        # Generator weights update.
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0 and do_log:
            loop.set_postfix(D_real=torch.sigmoid(d_real).mean().item(), D_fake=torch.sigmoid(d_fake).mean().item())

    return np.mean(gen_adv_losses), np.mean(gen_l1_losses), np.mean(disc_losses)


# Load train and test set of UnitoPatho.
def load_dataset_UTP(transform_train, transform_test):
    path = '../data/unitopath-public/800'
    mask_dir = "generated_torchstain" if config.HE_NORM else "generated"
    path_masks = f"../data/unitopath-public/{mask_dir}"

    # training set
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    df = df[df.grade >= 0].copy()
    train_dataset = UnitopathoMasks(df, T=transform_train, path=path, target='grade', path_masks=path_masks, train=True,
                                    device=torch.cuda.current_device())

    # test set
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    df = df[df.grade >= 0].copy()
    test_dataset = UnitopathoMasks(df, T=transform_test, path=path, target='grade', path_masks=path_masks, train=False,
                                   device=torch.cuda.current_device())

    return train_dataset, test_dataset