import os
import tarfile
import gdown
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PanNuke(Dataset):
    """
    Dataset must have two directories: train, test
    train and test directory must have other two subdirectories: data, masks
    """
    def __init__(self, root_dir="./data", train=True, transform=None, download=False):
        self.train = train
        self.transform = transform
        self.root_dir = root_dir
        if download:
            if os.path.exists(self.root_dir + "/CancerInstance"):
                print("Dataset already downloaded.")
            else:
                self.download()
            self.root_dir += "/CancerInstance"

        self.root_dir += ("/train" if self.train else "/test")
        self.images = sorted(os.listdir(f"{self.root_dir}/data"))
        self.masks = sorted(os.listdir(f"{self.root_dir}/masks"))

        if len(self.images) != len(self.masks):
            raise ValueError('Number of images not equal to number of masks.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]

        # load image from file
        img_path = os.path.join(f"{self.root_dir}/data", image)
        image = np.array(Image.open(img_path))  # (H, W, 3)

        # load mask from file
        mask_path = os.path.join(f"{self.root_dir}/masks", f"{mask.split('.')[0]}.npy")
        mask = np.load(mask_path).astype('float')  # (H, W, 6)

        if mask.sum() == 0:  # if the mask is empty
            mask += np.array([0, 0, 0, 0, 0, 1])  # put all background

        # transformation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask.argmax(axis=2))
            image, mask = transformed["image"], transformed["mask"]  # (3, H, W), (H, W)
            # make mask one_hot
            mask = nn.functional.one_hot(mask.long(), num_classes=len(PanNuke.labels())).float()  # (H, W, 6)
        else:
            image = torch.Tensor(image).permute(2, 0, 1)  # (3, H, W)
            image /= 255.  # normalize to [0, 1]
            mask = torch.Tensor(mask)

        # make mask of shape (6, H, W)
        mask = mask.permute(2, 0, 1)

        return {"image": image, "mask": mask}

    @staticmethod
    def labels():
        return ['Neoplastic cells', 'Inflammatory', 'Connective/Soft tissue cells', 'Dead Cells', 'Epithelial',
                'Background']

    @staticmethod
    def get_color_map():
        colors = ['b', 'g', 'r', 'c', 'm', 'w']
        labels = PanNuke.labels()
        return {matplotlib.colors.to_rgb(color): label for color, label in zip(colors, labels)}

    @staticmethod
    def get_img_mask(mask):
        """
            Get image of the mask.
            mask shape: [num_classes, H, W]
            :returns image of the mask with shape (3, H, W)
        """
        colors = list(PanNuke.get_color_map().keys())
        labels = PanNuke.labels()

        img_mask = np.empty((mask.shape[1], mask.shape[2], 3))
        mask = mask.argmax(dim=0)

        # assing to each label a color
        for idx_lbl in range(len(labels)):
            img_mask[mask == idx_lbl] = colors[idx_lbl]

        return torch.from_numpy(img_mask).permute(2, 0, 1)

    def download(self):
        # Fetch data from Google Drive
        # URL for the dataset
        url = "https://drive.google.com/uc?id=1_R3jCpMoNBA-vOkd_NJcHamZsv8E3v7Z"  # 0.9-0.1
        # url = "https://drive.google.com/uc?id=1cR4FdnoVznh8ZXmAu6AZzbylfYouKRj1"  # 0.7-0.3

        # Path to download the dataset to
        download_path = f'{self.root_dir}/cancer_instance.tar.xz'

        # Create required directories
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Download the dataset from google drive
        gdown.download(url, download_path, quiet=False)
        print('Downloaded!')

        print('Unzipping...')
        with tarfile.open(download_path) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root_dir)
        os.remove(download_path)
        print('Done!')


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(35, 35))

    for i, (name, imgs) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        plt.title(name)
        plt.imshow(torchvision.utils.make_grid(imgs, nrow=3).permute(1, 2, 0))

    # put those patched as legend-handles into the legend
    patches = [mpatches.Patch(color=color, label=label) for color, label in PanNuke.get_color_map().items()]
    plt.legend(handles=patches, bbox_to_anchor=(-1, -0.5), borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    dataset = PanNuke(download=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    images, masks = batch
    print(images.shape, masks.shape)
    visualize(
        Images=images,
        Masks=[PanNuke.get_img_mask(mask) for mask in masks],
    )
