# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.datasets
from torch.utils.data.dataloader import DataLoader


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
train_dataset = torchlanguage.utils.CrossValidation(dataset)
test_dataset = torchlanguage.utils.CrossValidation(dataset, train=False)

# Data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# For each fold
for k in range(10):
    print(u"Training")
    # Training set
    for data in train_dataset:
        # Inputs and outputs
        inputs, label = data
        print(label[0])
    # end for
    print(u"")
    print(u"Test")
    # Test set
    for data in test_dataset:
        # Inputs and outputs
        inputs, label = data
        print(label[0])
    # end for
    print(u"")
    print(u"")
    # Next fold
    train_dataset.next_fold()
    test_dataset.next_fold()
# end for
