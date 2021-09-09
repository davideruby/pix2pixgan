# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
import albumentations as albume
import config
import train_utils
import utils
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset.pannuke import PanNuke
from discriminator_model import Discriminator
from generator_model import Generator


transform_training = albume.Compose(
    [
        albume.Flip(p=0.75),
        albume.RandomRotate90(p=0.75),
        albume.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
transform_test = albume.Compose(
    [
        albume.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
WANDB_PROJECT_NAME = "pix2pixgan"


def main(gpu):
    # DDP
    print(f"#{gpu} started", flush=True)
    world_size = config.NGPU * config.NUM_NODES
    nr = 0  # it is the rank of the current node. Now we use only one node
    rank = nr * config.NGPU + gpu
    utils.setup_ddp(rank, world_size)
    torch.cuda.set_device(gpu)
    is_master = rank == 0
    do_wandb_log = config.LOG_WANDB and is_master  # only master logs
    if do_wandb_log:
        train_utils.wandb_init(config.WANDB_KEY_LOGIN, WANDB_PROJECT_NAME)

    # Models
    num_classes = len(PanNuke.labels())
    disc = Discriminator(in_channels=3 + num_classes).cuda(gpu)
    gen = Generator(in_channels=num_classes, features=64).cuda(gpu)
    # Use SynchBatchNorm for Multi-GPU trainings
    disc = nn.SyncBatchNorm.convert_sync_batchnorm(disc)
    gen = nn.SyncBatchNorm.convert_sync_batchnorm(gen)
    if do_wandb_log:
        print(disc)
        print(gen)

    # Weight initialization
    disc.apply(utils.init_weights)
    gen.apply(utils.init_weights)

    # DDP
    disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu])
    gen = nn.parallel.DistributedDataParallel(gen, device_ids=[gpu])

    # optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.ADAM_BETA1, config.ADAM_BETA2))

    # losses
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Load checkpoints from wandb
    if config.LOAD_MODEL:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        wandb_run_path = "daviderubi/pix2pixgan/1l0hnnnn"  # The wandb run is daviderubi/pix2pixgan/upbeat-river-42
        train_utils.wandb_load_model(wandb_run_path, "disc.pth", disc, opt_disc, config.LEARNING_RATE, map_location)
        train_utils.wandb_load_model(wandb_run_path, "gen.pth", gen, opt_gen, config.LEARNING_RATE, map_location)

    # train loader
    train_dataset = PanNuke(train=True, transform=transform_training, download=True)
    # DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                               num_workers=config.NUM_WORKERS, sampler=train_sampler)

    # test loader
    test_dataset = PanNuke(train=False, transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # grad_scaler
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if do_wandb_log:
        # Get some images from testloader. Every epoch we will log the generated images for this batch on wandb.
        test_batch_im, test_batch_masks = train_utils.wandb_get_images_to_log(test_loader)
        img_masks_test = [PanNuke.get_img_mask(mask) for mask in test_batch_masks]
        wandb.log({"Reals": wandb.Image(torchvision.utils.make_grid(test_batch_im), caption="Reals"),
                   "Masks": wandb.Image(torchvision.utils.make_grid(img_masks_test), caption="Masks")})

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        g_adv_loss, g_l1_loss, d_loss = train_utils.train_epoch(disc, gen, train_loader, opt_disc, opt_gen, l1_loss,
                                                                bce, g_scaler, d_scaler, gpu)

        if config.SAVE_MODEL and (epoch + 1) % 5 == 0 and is_master:
            # Save checkpoint.
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=epoch + 1)
            utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=epoch + 1)
            if do_wandb_log:
                wandb.save(config.CHECKPOINT_GEN)
                wandb.save(config.CHECKPOINT_DISC)

        if do_wandb_log:
            # Log generated images after the epoch.
            train_utils.wandb_log_epoch(gen, test_batch_masks, g_adv_loss, g_l1_loss, d_loss)

    # save gen and disc models
    if is_master:
        utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN, epoch=config.NUM_EPOCHS)
        utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC, epoch=config.NUM_EPOCHS)

    if do_wandb_log:
        train_utils.wandb_log_generated_images(gen, test_loader)
        wandb.finish()

    torch.distributed.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"Working on {config.DEVICE} device.")
    if "cuda" in str(config.DEVICE):
        for idx in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(idx))
    PanNuke(download=True)
    # DistributedDataParallel
    mp.spawn(main, nprocs=config.NGPU, args=())
