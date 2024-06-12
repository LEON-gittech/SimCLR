from ast import Str
import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import json
import random
import cv2 as cv

class Face(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        data_json = "/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/valid_face_2million.json"
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        with open(data_json, "r") as f:
            self.data_json = json.loads(f.read())
        self.data = []
        for d in self.data_json:
            self.data.append(f"{d['room_id']}/{d['object1']}")
            self.data.append(f"{d['room_id']}/{d['object2']}")
    
    def __getitem__(self, index: int) -> Any:
        d = self.data[index]
        img_path = random.choice(os.listdir(f"{self.root}/{d}"))
        img = Image.open(f"{self.root}/{d}/{img_path}")
        img = self.transform(img)
        return img, index
    
    def __len__(self) -> int:
        return len(self.data)
