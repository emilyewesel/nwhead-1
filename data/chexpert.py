import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset

class ChexpertDataset(Dataset):
    def __init__(self, csv_file, train_base_path, test_base_path, transform=None, train=True, inject_underdiagnosis_bias=False, train_class = "Edema"):
        self.df = pd.read_csv(csv_file)

        if train_class == "No Finding":
            self.df["No Finding"].fillna(0, inplace=True)
        else:
            self.df = self.df[self.df[train_class].isin([0, 1])]
        
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        self.df.dropna(subset=["Sex"], inplace=True)
        self.df = self.df[self.df.iloc[:, 1].isin(["Female", "Male"])]
        if train and inject_underdiagnosis_bias:
            female_indices = self.df[(self.df["Sex"] == "Female") & (self.df[train_class] == 1)].index
            num_female_samples = len(female_indices)
            num_samples_to_convert = int(0.25 * num_female_samples)
            indices_to_convert = np.random.choice(female_indices, num_samples_to_convert, replace=False)
            self.df.loc[indices_to_convert, train_class] = 0
        self.base_path = train_base_path if train else test_base_path
        self.transform = transform
        self.targets = torch.tensor(self.df[train_class].values, dtype=torch.long)  
        self.genders = self.df.iloc[:, 1].dropna().map({'Female': 1, 'Male': 0}).astype(int).tolist()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0].split('/', 1)[-1]
        img_name = os.path.join(self.base_path, img_name)
        image = Image.open(img_name).convert('RGB')  

        label = self.targets[idx]
        gender = self.genders[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, gender, img_name
    
    def compute_class_weights(self):
        class_counts = Counter(self.targets.numpy())
        total_samples = sum(class_counts.values())
        class_weights = [total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))]
        sum_weights = sum(class_weights)
        class_weights = [weight / sum_weights for weight in class_weights]
        return torch.tensor(class_weights)
    def compute_class_weights2(self):
        class_counts_male = Counter()
        class_counts_female = Counter()

        for label, gender in zip(self.targets.numpy(), self.genders):
            if gender == 0:  # Male
                class_counts_male[label] += 1
            else:  # Female
                class_counts_female[label] += 1

        return {
            'male': {
                'positive': class_counts_male[1],
                'negative': class_counts_male[0]
            },
            'female': {
                'positive': class_counts_female[1],
                'negative': class_counts_female[0]
            }
        }