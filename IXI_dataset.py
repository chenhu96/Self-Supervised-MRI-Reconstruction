import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import normalize_zero_to_one


class IXIData(Dataset):
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path, sample_rate):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        self.sample_rate = sample_rate

        self.examples = []
        files = list(pathlib.Path(self.data_path).iterdir())
        start_id, end_id = 0, 120
        for file in sorted(files):
            self.examples += [(file, slice_id) for slice_id in range(start_id, end_id)]
        if self.sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * self.sample_rate)
            self.examples = self.examples[:num_examples]

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        file, slice_id = self.examples[item]
        data = nib.load(file)
        label = data.dataobj[..., slice_id]
        label = normalize_zero_to_one(label, eps=1e-6)
        label = torch.from_numpy(label).unsqueeze(-1).float()
        return label, self.mask_under, self.mask_net_up, self.mask_net_down, file.name, slice_id
