from torch.utils.data import Dataset
import json
import os
import random
from torchvision.transforms import transforms
import cv2 as cv
import torch

class FaceDataset(Dataset):
    def __init__(self, data_path: str, data_json="/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/valid_face.json") -> None:
        self.root = data_path
        with open(data_json, "r") as f:
            self.data = json.loads(f.read())
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.resize = transforms.Resize((224,224))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.resize, 
            self.normalize
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        d = self.data[index]
        img1_path = random.choice(os.listdir(f'{self.root}/{d["room_id"]}/{d["object1"]}'))
        img2_path = random.choice(os.listdir(f'{self.root}/{d["room_id"]}/{d["object2"]}'))
        img1 = cv.imread(f'{self.root}/{d["room_id"]}/{d["object1"]}/{img1_path}')
        img2 = cv.imread(f'{self.root}/{d["room_id"]}/{d["object2"]}/{img2_path}')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return [img1,img2], [index,index]