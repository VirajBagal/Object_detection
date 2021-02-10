import cv2
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from utils import get_train_transform, get_valid_transform

def get_dataset(args):
    train_df = pd.read_csv(f'{args.dir_input}/train.csv')
    train_df[['x', 'y', 'w', 'h']] = pd.DataFrame(train_df['bbox'].apply(lambda x: eval(x)).tolist(), index = train_df.index)
    image_ids = sorted(train_df['image_id'].unique())
    index = 655
    valid_ids = image_ids[-index:]
    train_ids = image_ids[:-index]
    
    
    if args.debug:
        index = int(0.1*index)
        valid_ids = image_ids[-index:]
        train_ids = image_ids[:4*index]


    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    print(valid_df.shape)
    print(train_df.shape)

    return train_df, valid_df

def get_dataloaders(args, train_df, valid_df):

    train_dataset = WheatDataset(train_df, args.dir_train, get_train_transform())
    valid_dataset = WheatDataset(valid_df, args.dir_train, get_valid_transform())


    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn = collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn = collate_fn
    )
    
    return train_data_loader, valid_data_loader
    
    
def collate_fn(batch):
    return tuple(zip(*batch))


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.tensor(sample['bboxes'])

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]