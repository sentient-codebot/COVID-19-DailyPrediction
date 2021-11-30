from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image

class COVID19Daily(Dataset):
    def __init__(self, csv_file, segment_length):
        self.df = pd.read_csv(csv_file)
        self.segment_length = segment_length

    def __len__(self):
        return len(self.df)-self.segment_length # reserve the last as the last LABEL

    def __getitem__(self, idx):
        ''' idx: starting index '''
        past_seq = self.df.iloc[idx:idx+self.segment_length, -1].to_numpy()
        present = self.df.iloc[idx+self.segment_length, -1] # series, already numpy (?)

        return torch.tensor(past_seq).float(), torch.tensor(present).reshape(1).float()

class COVID19Average(Dataset):
    '''
    x: past p days
    y: average of future 7 seven days (including today)
    '''
    def __init__(self, csv_file, segment_length):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.segment_length = segment_length

    def __len__(self):
        return len(self.df)-self.segment_length-6 # reserve the last as the last LABEL

    def __getitem__(self, idx):
        ''' idx: starting index '''
        past_seq = self.df.iloc[idx:idx+self.segment_length, -1].to_numpy()
        pre_2_future = self.df.iloc[idx+self.segment_length:idx+self.segment_length+7, -1].to_numpy() # series, already numpy (?)
        average_future = np.mean(pre_2_future)

        return torch.tensor(past_seq).float(), torch.tensor(average_future).reshape(1).float()

def main():
    filename = "data/COVID-19_aantallen_nationale_per_dag.csv"
    dataset = COVID19Average(filename, 10)
    get = next(iter(dataset))
    print(f"DATASET returned item: {get[0].shape}, {get[1].shape}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    get = next(iter(dataloader))
    print(f"DATALOADER returned item: {get[0].shape}, {get[1].shape}")


if __name__ == "__main__":
    main()
