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

# Data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# For each batch
for data in data_loader:
    # Inputs and outputs
    inputs, label = data
    print(inputs)
    print(label)
    print(label[0])
    print(type(label))
# end for
