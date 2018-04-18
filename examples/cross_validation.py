# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.datasets
from torch.utils.data.dataloader import DataLoader


# Fold
k = 5

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Token(),
    torchlanguage.transforms.ToIndex(start_ix=1)
])

# Load from directory
dataset = torchlanguage.datasets.FileDirectory(
    download=True,
    download_url="http://www.nilsschaetti.com/datasets/sf1.zip",
    transform=transformer
)

# Cross validation
cross_val_dataset = {'train': torchlanguage.utils.CrossValidation(dataset, k=k),
                     'test': torchlanguage.utils.CrossValidation(dataset, k=k, train=False)}

# Data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# For each fold
for k in range(k):
    # Training set
    for data in cross_val_dataset['train']:
        # Inputs and outputs
        inputs, label = data
        print(label)
    # end for

    # Test set
    for data in cross_val_dataset['test']:
        # Inputs and outputs
        inputs, label = data
        print(label)
    # end for

    # Next fold
    cross_val_dataset['train'].next_fold()
    cross_val_dataset['test'].next_fold()
# end for
