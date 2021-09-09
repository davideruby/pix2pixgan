import os
import torchvision
from dataset.unitopatho import UTP
from dataset.pannuke import PanNuke
import torch


class UnitopathoMasks(UTP):
    def __init__(self, df, T, path, target, path_masks, train=True, device="cpu", subsample=-1, gray=False, mock=False):
        # super.__init() is called with T=None because we have to transform both image and mask together:
        # so when we call super().__get_item() we get the image not transformed
        super().__init__(df, None, path, target, subsample, gray, mock)
        self.transform = T
        self.path_masks = path_masks
        self.train = train
        self.device = device

    def read_mask(self, image_id):
        filename = image_id.split('/')[-1].strip("'")
        train = "training" if self.train else "test"
        path = os.path.join(self.path_masks, train, "masks", filename)
        mask = torchvision.io.decode_png(torchvision.io.read_file(path)).to(self.device)
        mask = mask / 255.  # normalize to [0, 1]

        # convert mask from image (h, w, 3) to one hot (h, w, classes)
        mask = self.convert_mask_to_one_hot(mask.permute(1, 2, 0))  # H, W, classes

        return mask

    # https://www.spacefish.biz/2020/11/rgb-segmentation-masks-to-classes-in-tensorflow/
    def convert_mask_to_one_hot(self, mask):
        """
        :param mask: h, w, 3
        :return: h, w, classes
        """
        def round(tensor, n_digits):  # round tensor to n_digits decimals
            return torch.round(tensor * 10 ** n_digits) / (10 ** n_digits)

        mask = round(mask, 2)

        colors = [torch.Tensor(color).to(self.device) for color in PanNuke.get_color_map().keys()]
        one_hot_map = []
        for color in colors:
            class_map = torch.all(torch.eq(mask, color), dim=-1)
            one_hot_map.append(class_map)

        one_hot_map = torch.stack(one_hot_map, dim=-1)
        return one_hot_map.float()

    def __getitem__(self, index):
        # get image
        image, target, image_id = super().__getitem__(index)
        image = torch.from_numpy(image).float().to(self.device)  # H, W, 3
        image /= 255.  # normalize to [0, 1]
        image = image.permute(2, 0, 1)  # 3, H, W

        # get mask
        mask = self.read_mask(image_id)  # H, W, classes
        mask = mask.permute(2, 0, 1)  # classes, H, W

        # transformation
        if self.transform is not None:
            # do transformations and normalize image to [-1, 1]
            image, mask = self.do_transformations(image, mask, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {"image": image, "mask": mask, "target": target, "image_id": image_id}

    def do_transformations(self, image, mask, mean=None, std=None):
        """
        It transforms both image and mask according to self.transform.
        It also normalizes the image with mean and standard deviation if provided.
        :return: transformed img and mask
        """
        img_ch = image.size()[0]  # 3 for RGB

        # transform both image and mask together
        image_mask = self.transform(torch.cat([image, mask], dim=0))

        there_is_five_crop = any(isinstance(tr, torchvision.transforms.FiveCrop) for tr in self.transform.transforms)
        if there_is_five_crop:
            image, mask = image_mask[:, :img_ch, ...], image_mask[:, img_ch:, ...]
        else:
            image, mask = image_mask[:img_ch, ...], image_mask[img_ch:, ...]

        if (mean is not None) and (std is not None):
            image = torchvision.transforms.Normalize(mean=mean, std=std)(image)

        return image, mask
