import os
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Optional, Callable, Any, Tuple

from torch.utils.data import DataLoader


class ZeroWastedataset(VisionDataset):
    """Pytorch implementation of the ZeroWaste dataset."""

    def __init__(
            self,
            root: str,
            split: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super().__init__(
            root=os.path.join(root, split),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform)
        self.sem_seg_root = os.path.join(self.root, "sem_seg")
        self.img_root = os.path.join(self.root, "data")
        self.imgs = os.listdir(self.img_root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_name = self.imgs[index]
        image = Image.open(os.path.join(self.img_root, img_name)).convert('RGB')
        target = Image.open(os.path.join(self.sem_seg_root, img_name)).convert('RGB')
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)


# ZeroWaste palette
PALETTE = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]

if __name__ == '__main__':
    root_path = "D:/Documents/dataset/visda2022"
    zerowaste_v1 = {data_set: ZeroWastedataset(root=os.path.join(root_path, 'zerowaste-f'), split=data_set)
                    for data_set in ['train', 'val', 'test']}

    dataloader = {data_set: DataLoader(zerowaste_v1[data_set])
                  for data_set in ['train', 'val', 'test']}

    for split in ['train', 'val', 'test']:
        for idx, (img, target) in enumerate(zip(dataloader[split])):
            color_target = torch.zeros_like(target)

