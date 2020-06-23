import os
import numpy as np
import torch
from PIL import Image
from glob import glob
import re
import csv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # load all image files
        self.ann_file = os.path.join(root, 'labels.csv')
        self.ann = load_labels(self.ann_file)
        self.n_imgs = len(self.ann)

    def __getitem__(self, idx):
        # load images and masks
        seq = self.ann[idx][0]
        frame = self.ann[idx][1]

        img_path = self.root + 'seq' + seq + '/img' + frame + '.jpg'
        img = Image.open(img_path).convert('RGB')

        # get bounding box coordinates for each mask
        boxes = [x for x in self.ann[idx][2]]
        n_objs = len(boxes)
        if n_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((n_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((n_objs,), dtype=torch.int64)        

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, (seq, frame)

    def __len__(self):
        return self.n_imgs

class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        pattern = re.compile('img(.*).jpg')
        self.imgs = list()
        sequences = glob(os.path.join(root, "seq*"))
        for seq in sequences:
            seq_num = seq[-3::]
            img_files = glob(os.path.join(seq, "img*.jpg"))
            for img_f in img_files:

                img_num = pattern.search(os.path.basename(img_f)).group(1)
                self.imgs.append((img_f, seq_num, img_num))

    def __getitem__(self, idx):
        img_file = self.imgs[idx][0]
        seq = self.imgs[idx][1]
        frame = self.imgs[idx][2]

        # load image
        img = Image.open(img_file).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, seq, frame

    def __len__(self):
        return len(self.imgs)

def load_labels(path_to_csv):
    labels = []
    with open(path_to_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i==0: # header
                continue
            if len(row) == 3:
                boxes = eval(row[2])
            else:
                boxes = []
            labels.append([row[0], row[1], boxes])
    """
    with open(path_to_csv, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.split(';')            
            if len(line)==3:
                boxes = eval(line[2])
            else:
                boxes = []
            labels.append([line[0], line[1], boxes])

            box = line[2][2:-3]
                for char in '()':
                    box = box.replace(char,'')            
                box = box.split(',')
                bboxs = []
                for bb in range(len(box)//4):
                    bbox = [int(x) for x in box[4*bb:4*bb+4]]
                    bboxs.append(bbox)
    """
    return labels
