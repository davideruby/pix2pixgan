import torchvision
from dataset.unitopatho import UTP
from dataset.pannuke import CancerInstanceDataset
import torch


class UTP_Masks(UTP):
    def __init__(self, df, T, path, target, path_masks, train=True, device="cpu", subsample=-1, gray=False, mock=False):
        # super.__init() is called with T=None because we have to transform both image and mask together:
        # so when we call super().__get_item() we get the image not transformed
        super().__init__(df, None, path, target, subsample, gray, mock)
        self.transform = T
        self.path_masks = path_masks
        self.train = train
        self.device = device

    def read_mask(self, image_id):
        filename = image_id.split('/')[-1].strip("'").strip(".png")
        train = "training" if self.train else "test"
        path = f"{self.path_masks}/{train}/masks/{filename}.png"
        mask = torchvision.io.decode_png(torchvision.io.read_file(path)).to(self.device)
        mask = mask.permute(1, 2, 0)
        mask = mask / 255

        # convert mask from image (h, w, 3) to one hot (h, w, classes)
        mask = self.convert_mask_from_img_to_one_hot(mask)  # H, W, classes

        return mask

    # https://www.spacefish.biz/2020/11/rgb-segmentation-masks-to-classes-in-tensorflow/
    def convert_mask_from_img_to_one_hot(self, mask):
        """
        :param mask: h, w, 3
        :return: h, w, 6
        """
        def round(tensor, n_digits):  # round tensor to n_digits decimals
            return torch.round(tensor * 10 ** n_digits) / (10 ** n_digits)

        mask = round(mask, 2)

        colors = [torch.Tensor(color).to(self.device) for color in CancerInstanceDataset.get_color_map().keys()]
        one_hot_map = []
        for color in colors:
            class_map = torch.all(torch.eq(mask, color), dim=-1)
            one_hot_map.append(class_map)

        one_hot_map = torch.stack(one_hot_map, dim=-1)
        return one_hot_map.float()

    def __getitem__(self, index):
        image, target, image_id = super().__getitem__(index)

        image = torch.from_numpy(image).to(self.device)  # H, W, 3
        mask = self.read_mask(image_id)  # H, W, classes

        # transformation
        if self.transform is not None:
            image, mask = self.do_transformations(image, mask)
        else:
            image = torch.Tensor(image).permute(2, 0, 1)  # (3, H, W)
            image /= 255.  # normalize to [0, 1]
            mask = torch.Tensor(mask)
            # make mask of shape (classes, H, W)
            mask = mask.permute(2, 0, 1)

        return {"image": image, "mask": mask, "target": target, "image_id": image_id}

    def do_transformations(self, image, mask):
        """
        :param image: H, W, 3
        :param mask: H, W, classes
        :return: img (3, H, W), mask (classes, H, W)
        """
        image = image.permute(2, 0, 1)  # 3, H, W
        mask = mask.permute(2, 0, 1)  # classes, H, W
        ch = image.size()[0]  # 3 for RGB
        # transform both image and mask together
        image_mask = self.transform(torch.cat((image, mask), dim=0))

        there_is_five_crop = any(isinstance(tr, torchvision.transforms.FiveCrop) for tr in self.transform.transforms)
        if there_is_five_crop:
            image, mask = image_mask[:, :ch, ...], image_mask[:, ch:, ...]
        else:
            image, mask = image_mask[:ch, ...], image_mask[ch:, ...]

        # normalize image in [-1, 1]
        image = ((image / 255) - 0.5) / 0.5

        return image, mask

# def convert_mask_from_img_to_one_hot(self, img):
#     """
#     :param img: image of the mask. Image must be normalized in [0, 1]. Shape: (H, W, 3)
#     :return: a numpy array of the mask in one hot encoding. Shape: (H, W, classes)
#     """
#     img = np.round(img, 2)
#     num_classes = len(CancerInstanceDataset.labels())
#     one_hot = np.empty((img.shape[0], img.shape[1], num_classes))
#     colors = CancerInstanceDataset.get_color_map().keys()
#
#     for idx_class, color in enumerate(colors):
#         idxs = np.all(img[:, :] == color, axis=2)  # indices where the img is equal to the color of the class
#         one_hot[:, :, idx_class][idxs] = 1
#     return one_hot
