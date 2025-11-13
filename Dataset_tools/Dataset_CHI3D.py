import numpy as np
import os
from torch.utils.data import Dataset
import torch
import json
from torch.utils.data import DataLoader

class Datasets(Dataset):

    def __init__(self, opt, mode):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing and validation
        :param sample_rate:
        """

        self.path_to_data = "/home/s01040/ST-MoE/Dataset/CHI3D/train"
        self.mode = mode
        self.frame_in = opt.frame_in
        self.frame_out = opt.frame_out
        self.sample_rate = 2
        self.seq_len = self.frame_in + self.frame_out


        subs = [[2, 3], [4]]

        self.dimensions_to_used = np.array([0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 9])
        self.dim_used = np.concatenate((self.dimensions_to_used * 3, self.dimensions_to_used * 3 + 1, self.dimensions_to_used * 3 + 2))
        self.in_features = len(self.dim_used)

        subs = subs[mode]

        all_data = None

        for i in subs:
            mod_dir = '{0}/s0{1}/{2}'.format(self.path_to_data, i, "joints3d_25")
            # e.g. '../Dataset/CI3D/chi3d_train/train/s02/joints3d_25'
            _, _, filenames = self.module_name(mod_dir)
            filenames.sort()
            mod_data = None
            for j in range(len(filenames)):
                curr_file = os.path.join(mod_dir, filenames[j])
                # curr_file = os.path.normpath(os.path.join(mod_dir, filenames[j]))
                # e.g. '../Dataset/CI3D/chi3d_train/train/s02/joints3d_25/Grab 1.json'
                data = self.json_read_CHI(curr_file)
                num_frames = data.shape[1]

                data = data[:, np.arange(0, num_frames, self.sample_rate), :, :]
                num_frames = data.shape[1]

                fs = np.arange(0, num_frames - self.seq_len + 1)  
                fs_sel = fs  
                for k in np.arange(0, self.seq_len - 1):  
                    fs_sel = np.vstack((fs_sel, fs + k + 1))  

                fs_sel = fs_sel.transpose()
                seq_sel = data[:, fs_sel, :, :]

                if mod_data is None:
                    mod_data = seq_sel
                else:
                    mod_data = np.concatenate((mod_data, seq_sel), axis=1)

            if all_data is None:
                all_data = mod_data
            else:
                all_data = np.concatenate((all_data, mod_data), axis=1)

        all_data = all_data[:, :, :, self.dimensions_to_used, :]
        num_person = all_data.shape[0]
        num_frames = all_data.shape[1]

        all_data = all_data.reshape(num_person, num_frames, opt.seq_len, -1)
        all_data = all_data.transpose(1, 0, 3, 2)  
 
        pad_idx = np.repeat([self.frame_in - 1], self.frame_out) 
        i_idx = np.append(np.arange(0, self.frame_in), pad_idx)

        input_dct_seq = all_data[:, :, :, i_idx]

        output_dct_seq = all_data

        self.input_dct_seq = torch.from_numpy(input_dct_seq).float().cuda()
        self.output_dct_seq = torch.from_numpy(output_dct_seq).float().cuda()


    def module_name(self, dir):
        for root, dirnames, filenames in os.walk(dir):
            return root, dirnames, filenames
    
    def json_read_CHI(self,filename):
        f = open(filename, 'r')
        content = f.read()
        a = json.loads(content)
        data = a["joints3d_25"]
        data = np.array(data)
        return data

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item]