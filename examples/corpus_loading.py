# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.datasets
from torch.utils.data.dataloader import DataLoader


# Load from directory
dataset = torchlanguage.datasets.FileDirectory(download=True, download_url="http://www.nilsschaetti.com/datasets/sf1.zip")

# Data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# For each batch
for data in data_loader:
    # Inputs and outputs
    inputs, targets = data
    print(inputs)
    print(targets)
# end for
