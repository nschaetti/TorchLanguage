# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.datasets
from torch.utils.data.dataloader import DataLoader


# Fold
k = 10

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Token(),
    torchlanguage.transforms.ToIndex(start_ix=1)
])

# Load from directory
dataset = torchlanguage.datasets.ReutersC50Dataset(
    n_authors=15,
    download=True,
    transform=transformer
)

# Cross validation
cross_val_dataset = {'train': torchlanguage.utils.CrossValidation(dataset, k=k),
                     'test': torchlanguage.utils.CrossValidation(dataset, k=k, train=False)}

# Data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# For each fold
for k in range(k):
    # Log
    print(u"Fold {}".format(k))
    print(u"Training")

    # Count
    count = 0

    # Training set
    for data in cross_val_dataset['train']:
        # Inputs and outputs
        inputs, label, time_labels = data
        count += 1
    # end for

    # Log
    print(u"{} samples".format(count))
    print(u"Test")

    # Reinit
    count = 0

    # Test set
    for data in cross_val_dataset['test']:
        # Inputs and outputs
        inputs, label, time_labels = data
        count += 1
    # end for

    # Space
    print(u"{} samples".format(count))
    print(u"")

    # Next fold
    cross_val_dataset['train'].next_fold()
    cross_val_dataset['test'].next_fold()
# end for
