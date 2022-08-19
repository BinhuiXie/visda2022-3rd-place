import os
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Optional, Callable, Any, Tuple


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