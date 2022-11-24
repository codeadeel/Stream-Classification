#!/usr/bin/env python3

# %%
# NOTE
# ====
# This file is responsible to download trained resources from Google Drive. Please run this script after cloning this repository to get trained resources.
#
# To install gdown, please visit following repository.
#		https://github.com/wkentaro/gdown

# %%
# Importing Libraries
import os
import gdown

# %%
# Download Configuration

dlinks = {
    'ffmpeg': 'https://drive.google.com/file/d/1mZ-Pst32ZWrLKvFi-pki_kSOFsPVSJOm/view?usp=sharing',
    'youtube-dl' : 'https://drive.google.com/file/d/1y8-9IUxz3s6D9Gw4OjrvhBy7MhRjzOpR/view?usp=sharing',
    'convnext_tiny-983f1562.pth' : 'https://drive.google.com/file/d/1oErfEfqfb5llR7JW2qgyVh9d3ZrQ6r6l/view?usp=sharing',
    'convnext.model' : 'https://drive.google.com/file/d/14aAEwiOs6Me1yQ2Ff7xiNsNJz2WM_IQT/view?usp=sharing',
    'OHE.labels' : 'https://drive.google.com/file/d/1CazgVhZEzWaHffX_-FCb7jIHJptDTDHI/view?usp=sharing'
}

alternateDlinks = {
    'convnext_tiny-983f1562.pth' : 'https://drive.google.com/uc?id=1oErfEfqfb5llR7JW2qgyVh9d3ZrQ6r6l&confirm=t&uuid=8dcb6fa4-f763-4d81-a235-a80e6b93c891',
    'convnext.model' : 'https://drive.google.com/uc?id=14aAEwiOs6Me1yQ2Ff7xiNsNJz2WM_IQT&confirm=t&uuid=c8430e8f-6984-4f26-87cb-f26b8860cb49',
}

# %%
# Download Execution

download_addr = '/'.join(__file__.split('/')[:-1]) + '/Resources'

for keys, vals in dlinks.items():
    gdown.download(vals, f'{download_addr}/{keys}', quiet=False, fuzzy=True)
    if not os.path.exists(f'{download_addr}/{keys}'):
        gdown.download(alternateDlinks[keys], f'{download_addr}/{keys}', quiet=False)
