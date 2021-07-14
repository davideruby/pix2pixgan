from dataset.unitopatho import UTP
from dataset.pannuke import CancerInstanceDataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

class UTP_Masks(UTP):
    def __init__(self, df, T, path, target, path_masks, train=True, subsample=-1, gray=False, mock=False):
        # super.__init() is called with T=None because we have to transform both image and mask together:
        # so when we call super().__get_item() we get the image not transformed
        super().__init__(df, None, path, target, subsample, gray, mock)
        self.transform = T
        self.path_masks = path_masks
        self.train = train

    def read_mask(self, image_id):
        filename = image_id.split('/')[-1].strip("'").strip(".png")
        train = "training" if self.train else "test"
        path = f"{self.path_masks}/{train}/masks/{filename}.png"
        mask = np.array(Image.open(path)) / 255
        # convert mask from png to one hot numpy array
        mask = self.convert_mask_from_img_to_one_hot(mask)  # H, W, classes
        return mask

    def convert_mask_from_img_to_one_hot(self, img):
        """
        :param img: image of the mask. Image must be normalized in [0, 1]. Shape: (H, W, 3)
        :return: a numpy array of the mask in one hot encoding. Shape: (H, W, classes)
        """
        img = np.round(img, 2)
        num_classes = len(CancerInstanceDataset.labels())
        one_hot = np.empty((img.shape[0], img.shape[1], num_classes))
        colors = CancerInstanceDataset.get_color_map().keys()

        for idx_class, color in enumerate(colors):
            idxs = np.all(img[:, :] == color, axis=2)  # indices where the img is equal to the color of the class
            one_hot[:, :, idx_class][idxs] = 1
        return one_hot

    def __getitem__(self, index):
        image, target, image_id = super().__getitem__(index)
        mask = self.read_mask(image_id)  # H, W, classes

        # transformation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask.argmax(axis=2))
            image, mask = transformed["image"], transformed["mask"]  # (3, H, W), (H, W)
            # make mask one_hot
            mask = nn.functional.one_hot(mask.long(),
                                         num_classes=len(CancerInstanceDataset.labels())).float()  # (H, W, classes)
        else:
            image = torch.Tensor(image).permute(2, 0, 1)  # (3, H, W)
            image /= 255.  # normalize to [0, 1]
            mask = torch.Tensor(mask)

        # make mask of shape (classes, H, W)
        mask = mask.permute(2, 0, 1)

        return image, mask  # (3, H, W), (classes, H, W)




