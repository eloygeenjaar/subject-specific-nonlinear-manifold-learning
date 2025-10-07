import torch
import numpy as np
import pandas as pd
import nibabel as nb
import hcp_utils as hcp
from pathlib import Path
from sklearn import datasets
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from nilearn.image import resample_img
from nilearn import signal

# Reference: https://github.com/ericabusch/tphate_analysis_capsule/blob/main/code/utils.py
def apply_volume_ROI(ROI_nii, volume_image):
    # if the affine of the mask and volume don't match, resample thhe mask to the volume
    resampled_ROI = resample_img(ROI_nii, target_affine=volume_image.affine, target_shape=volume_image.shape[:-1]).get_fdata()
    resampled_ROI_mask = np.where(resampled_ROI > 0.95, 1, 0) # binarize it
    # now apply the mask to the data
    volume_image = volume_image.get_fdata()
    masked_img = volume_image[resampled_ROI_mask == 1, :]
    return masked_img

def apply_volume_ROI_nearest(ROI_nii, volume_image):
    # if the affine of the mask and volume don't match, resample thhe mask to the volume
    resampled_ROI = resample_img(ROI_nii, target_affine=volume_image.affine, target_shape=volume_image.shape[:-1], interpolation='nearest').get_fdata()
    # now apply the mask to the data
    volume_image = volume_image.get_fdata()
    masked_img = volume_image[resampled_ROI == 1, :]
    return masked_img


roi_dict = {
    'AUD': 'aud_early',
    'EV': 'early_visual',
    'LV': 'high_visual',
    'PMC': 'pmc_nn',
    'WholeBrain': 'WholeBrain',
    'DMN': 'default_mode',
    'CC': 'cognitive_control'
}

latent_dim_dict = {
    'Sherlock': {
        'aud_early': 41,
        'pmc_nn': 29,
        'early_visual': 30,
        'WholeBrain': 64
    },
    'Forrest': {
        'aud_early': 32,
        'default_mode': 32,
        'cognitive_control': 32,
        'WholeBrain': 64
    }
}

class SherlockData(Dataset):
    def __init__(self, data_type: str, roi_name: str, time_shuffle: bool, first_half=True):
        super().__init__()
        self.df = pd.read_csv('/data/users1/egeenjaar/subject-decomposition/data/SherlockBinary.csv', index_col=0)
        # We use shortened naming for the ROIs in the config
        roi_name = roi_dict[roi_name]
        self.roi_name = roi_name
        self.type_classes = ['IndoorOutdoor', 'MusicPresent']

        if roi_name == 'high_visual':
            roi = nb.load(f'/data/users1/egeenjaar/subject-decomposition/data/ROIs/{roi_name}.nii.gz')
        elif roi_name != 'WholeBrain':
            roi = nb.load(f'/data/users1/egeenjaar/subject-decomposition/data/ROIs/{roi_name}.nii')

        if not Path(f'/data/users1/egeenjaar/subject-decomposition/data/sherlock_{roi_name}.npy').is_file():
            movie_files = [f'/data/users1/egeenjaar/data/sherlock/movie_files/sherlock_movie_s{i}.nii' for i in range(1, 18) if i != 5]
            if roi_name != 'WholeBrain':
                self.movie_data = [apply_volume_ROI(roi, nb.load(movie_file)) for movie_file in movie_files]
            self.movie_data = np.stack(self.movie_data, axis=0)
            self.movie_data = torch.from_numpy(self.movie_data).float().permute(2, 0, 1)
            for i in range(self.movie_data.size(1)):
                mask = self.movie_data[:, i].std(0) != 0
                self.movie_data[:, i, mask] -= self.movie_data[:, i, mask].mean(0)
                self.movie_data[:, i, mask] /= self.movie_data[:, i, mask].std(0)
                self.movie_data[:, i, ~mask] = 0
            np.save(f'/data/users1/egeenjaar/subject-decomposition/data/sherlock_{roi_name}.npy', self.movie_data.numpy().astype(np.float16))
            self.movie_data = self.movie_data.half()
        else:
            self.movie_data = torch.from_numpy(
                np.load(f'/data/users1/egeenjaar/subject-decomposition/data/sherlock_{roi_name}.npy')).half()

        ix_TR = np.arange(self.df.shape[0])

        if time_shuffle:
            #ix_trainvalid, ix_test, y_trainvalid, y_test = train_test_split(
            #    ix_TR, self.df[self.type_classes].values, random_state=42, shuffle=False, stratify=self.df[self.type_classes].values, train_size=0.8)
            ix_trainvalid, ix_test, y_trainvalid, y_test = train_test_split(
                ix_TR, self.df[self.type_classes].values, random_state=42, shuffle=False, train_size=0.8)
            #ix_train, ix_valid, y_train, y_valid = train_test_split(
            #    ix_trainvalid, y_trainvalid, random_state=42, shuffle=False, stratify=y_trainvalid, train_size=0.9)
            ix_train, ix_valid, y_train, y_valid = train_test_split(
                ix_trainvalid, y_trainvalid, random_state=42, shuffle=False, train_size=0.9)
        else:
            # Take first half for training + validation and second half for test
            ix_trainvalid, ix_test, y_trainvalid, y_test = train_test_split(
                ix_TR, self.df[self.type_classes].values, random_state=42, shuffle=False, train_size=0.5)
            ix_train, ix_valid, y_train, y_valid = train_test_split(
                ix_trainvalid, y_trainvalid, random_state=42, shuffle=False, train_size=0.75)

        if data_type == 'train':
            self.data = self.movie_data[ix_train].half()
            self.y = torch.from_numpy(y_train).long()
        elif data_type == 'valid':
            self.data = self.movie_data[ix_valid].half()
            self.y = torch.from_numpy(y_valid).long()
        elif data_type == 'test':
            self.data = self.movie_data[ix_test].half()
            self.y = torch.from_numpy(y_test).long()

        self.num_timesteps = self.data.size(0)
        self.num_subjects = self.data.size(1)
        self.temporal_ixs = torch.arange(self.num_timesteps).unsqueeze(1).repeat(1, self.num_subjects).view(-1)
        self.data = self.data.view(-1, self.data.size(-1))
        self.y = self.y.unsqueeze(1).repeat(1, self.num_subjects, 1).view(-1, self.y.size(-1))
        self.subject_ixs = torch.arange(self.num_subjects).unsqueeze(0).repeat(self.num_timesteps, 1)
        self.subject_ixs = self.subject_ixs.view(-1)
        del self.movie_data

    def __len__(self):
        return self.data.size(0)
    
    @property
    def latent_dim(self):
        return latent_dim_dict['Sherlock'][self.roi_name]
    
    @property
    def data_size(self):
        return {
            'aud_early': 1018,
            'early_visual': 307,
            'pmc_nn': 481,
            'WholeBrain': 271633
        }[self.roi_name]

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs[ix]
        return x, subject_ix, temporal_ix, y

class ForrestData(Dataset):
    def __init__(self, data_type: str, roi_name: str, time_shuffle: bool, first_half=True):
        super().__init__()
        self.df = pd.read_csv('/data/users1/egeenjaar/subject-decomposition/data/forrest_movie_labels_coded_expanded.csv', index_col=0)
        self.df = self.df.iloc[:3541]
        # Make the labels go from 0 to
        self.df['FoT_coded'] = self.df['FoT_coded'] + 1
        self.type_classes = ['IoE_coded', 'ToD_coded', 'FoT_coded']

        # We use shortened naming for the ROIs in the config
        roi_name = roi_dict[roi_name]
        self.roi_name = roi_name

        if roi_name == 'high_visual':
            roi = nb.load(f'/data/users1/egeenjaar/subject-decomposition/data/ROIs/{roi_name}.nii.gz')
        elif roi_name != 'WholeBrain':
            roi = nb.load(f'/data/users1/egeenjaar/subject-decomposition/data/ROIs/{roi_name}.nii')

        if not Path(f'/data/users1/egeenjaar/subject-decomposition/data/forrest_{roi_name}.npy').is_file():
            if roi_name != 'WholeBrain':
                subject_paths = [Path(f'/data/users1/egeenjaar/data/forrest-fmri/linear_anatomical_alignment/sub-{str(i).zfill(2)}') for i in range(1, 21) if i not in [4, 10, 11, 13]]
                subjects = []
                for subject_path in subject_paths:
                    subject_string = subject_path.name
                    sessions = []
                    for i in range(1, 9):
                        fmri_file = subject_path / f'ses-forrestgump/func/{subject_string}_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7Tad_run-{str(i).zfill(2)}_bold.nii.gz'
                        session = apply_volume_ROI_nearest(roi, nb.load(fmri_file))
                        # Output is (num_voxels, num_timesteps)
                        session = session.T
                        session = signal.clean(session, detrend=True, standardize='zscore_sample', t_r=2.0)
                        # Based on: https://github.com/mjboos/forrest/blob/master/forrest/preprocessing.py
                        # And the original paper "...  while removing the last and first four volumes at each boundary connecting the runs."
                        # From: https://www.nature.com/articles/sdata20143
                        if i == 1:
                            # Shift the labels by two TRs (4-5s hemodynamic response)
                            session = session[2:-4]
                        elif i == 8:
                            session = session[4:]
                        else:
                            session = session[4:-4]
                        sessions.append(session)
                    # Reduce memory usage
                    subject_data = np.concatenate(sessions, axis=0).astype(np.float16)
                    print(subject_data.shape)
                    subjects.append(subject_data)
                self.movie_data = np.stack(subjects, axis=1)
                np.save(f'/data/users1/egeenjaar/subject-decomposition/data/forrest_{roi_name}.npy', self.movie_data)
                del subjects
                self.movie_data = torch.from_numpy(self.movie_data).half()
            else:
                subjects = []
                for i in range(16):
                    subject_data = np.load(f'/data/users1/egeenjaar/subject-decomposition/data/forrest_full/subject_{i}.npy', mmap_mode='r').astype(np.float16)
                    subjects.append(subject_data)
        else:
            self.movie_data = torch.from_numpy(np.load(f'/data/users1/egeenjaar/subject-decomposition/data/forrest_{roi_name}.npy')).half()

        ix_TR = np.arange(self.df.shape[0])

        if first_half:
            # Take first half for training + validation and second half for test
            ix_trainvalid, ix_test, y_trainvalid, y_test = train_test_split(
                ix_TR, self.df[self.type_classes].values, random_state=42, shuffle=False, train_size=0.5)
            ix_train, ix_valid, y_train, y_valid = train_test_split(
                ix_trainvalid, y_trainvalid, random_state=42, shuffle=False, train_size=0.75)
        else:
            # Take second half for training + validation and first half for test
            ix_test, ix_trainvalid, y_test, y_trainvalid = train_test_split(
                ix_TR, self.df[self.type_classes].values, random_state=42, shuffle=False, train_size=0.5)
            ix_train, ix_valid, y_train, y_valid = train_test_split(
                ix_trainvalid, y_trainvalid, random_state=42, shuffle=False, train_size=0.75)

        if roi_name == 'WholeBrain':
            if data_type == 'train':
                self.data = [subject[ix_train] for subject in subjects]
                self.y = torch.from_numpy(y_train).long()
            elif data_type == 'valid':
                self.data = [subject[ix_valid] for subject in subjects]
                self.y = torch.from_numpy(y_valid).long()
            elif data_type == 'test':
                self.data = [subject[ix_test] for subject in subjects]
                self.y = torch.from_numpy(y_test).long()

            self.num_timesteps = self.data[0].shape[0]
            self.num_subjects = len(self.data)
        else:
            if data_type == 'train':
                self.data = self.movie_data[ix_train].half()
                self.y = torch.from_numpy(y_train).long()
            elif data_type == 'valid':
                self.data = self.movie_data[ix_valid].half()
                self.y = torch.from_numpy(y_valid).long()
            elif data_type == 'test':
                self.data = self.movie_data[ix_test].half()
                self.y = torch.from_numpy(y_test).long()

            self.num_timesteps = self.data.size(0)
            self.num_subjects = self.data.size(1)
            self.data = self.data.view(-1, self.data.size(-1))
        self.temporal_ixs = torch.arange(self.num_timesteps).unsqueeze(1).repeat(1, self.num_subjects).view(-1)
        self.y = self.y.unsqueeze(1).repeat(1, self.num_subjects, 1).view(-1, self.y.size(-1))
        self.subject_ixs = torch.arange(self.num_subjects).unsqueeze(0).repeat(self.num_timesteps, 1)
        self.subject_ixs = self.subject_ixs.view(-1)

    def __len__(self):
        return self.num_timesteps * self.num_subjects

    @property
    def num_classes(self):
        return 1
    
    @property
    def latent_dim(self):
        return latent_dim_dict['Forrest'][self.roi_name]
    
    @property
    def data_size(self):
        return {
            'aud_early': 10211,
            'default_mode': 50144,
            'cognitive_control': 89745,
            'WholeBrain': 441100
        }[self.roi_name]

    def __getitem__(self, ix):
        # Select the data for the subject
        if self.roi_name == 'WholeBrain':
            x = torch.from_numpy(self.data[self.subject_ixs[ix]][self.temporal_ixs[ix]]).half()
        else:
            x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs[ix]
        return x, subject_ix, temporal_ix, y

class fBIRNSubjectData(Dataset):
    def __init__(self, data_type: str, roi_name: str, time_shuffle: bool):
        super().__init__()
        
        data_p = Path('/data/users1/egeenjaar/data/fbirn_temp')
        self.data = np.load(data_p / f'{data_type}.npz', allow_pickle=True)
        self.subjects = self.data['subjects']
        self.data = torch.from_numpy(self.data['data']).half()

        self.num_subjects = self.data.size(0)
        self.num_timesteps = self.data.size(1)
        self.temporal_ixs = torch.arange(self.num_timesteps)
        self.y = torch.zeros((self.num_subjects * self.num_timesteps, ))
        self.subject_ixs = torch.arange(self.num_subjects)
        self.num_classes = 2

    def __len__(self):
        return self.num_subjects
    
    @property
    def latent_dim(self):
        return 64
    
    @property
    def data_size(self):
        return 60409

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs
        return x, subject_ix, temporal_ix, y

    def index_subject(self, subject_ix):
        return self.data[subject_ix]

class MoonsData(Dataset):
    def __init__(self, data_type: str, roi_name: str, time_shuffle: bool):
        super().__init__()
        self.noise_level = 0.1
        self.num_samples = 1000
        X, y = datasets.make_moons(self.num_samples, noise=self.noise_level, shuffle=True, random_state=42)

        # Use music present because it has a better distribution
        x_trainvalid, x_test, y_trainvalid, y_test = train_test_split(
            X, y, random_state=42, shuffle=True, stratify=y, train_size=0.8)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_trainvalid, y_trainvalid, random_state=42, shuffle=True, stratify=y_trainvalid, train_size=0.9)

        if data_type == 'train':
            self.data = x_train
            self.y = y_train
        elif data_type == 'valid':
            self.data = x_valid
            self.y = y_valid
        elif data_type == 'test':
            self.data = x_test
            self.y = y_test

        self.data = torch.from_numpy(self.data).float()
        self.num_subjects = 1
        self.num_timesteps = self.data.size(0)
        self.y = torch.from_numpy(self.y).float().squeeze()
        self.subject_ixs = torch.ones(self.num_timesteps)
        self.temporal_ixs = torch.arange(self.num_timesteps)

    def __len__(self):
        return self.data.size(0)

    @property
    def num_classes(self):
        return 1
    
    @property
    def data_size(self):
        return 2

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs[ix]
        return x, subject_ix, temporal_ix, y

class SubjectMoonsData(Dataset):
    def __init__(self, data_type: str, roi_name: str, time_shuffle: bool, noise_level=0.1, num_subjects=128):
        super().__init__()
        self.noise_level = noise_level
        self.num_samples = 1000
        X, y = datasets.make_moons(self.num_samples, noise=self.noise_level, shuffle=True, random_state=42)
        self.X, self.Y = X, y

        # Use music present because it has a better distribution
        x_trainvalid, x_test, y_trainvalid, y_test = train_test_split(
            X, y, random_state=42, shuffle=True, stratify=y, train_size=0.8)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_trainvalid, y_trainvalid, random_state=42, shuffle=True, stratify=y_trainvalid, train_size=0.9)

        if data_type == 'train':
            self.orig_data = x_train
            self.orig_y = y_train
        elif data_type == 'valid':
            self.orig_data = x_valid
            self.orig_y = y_valid
        elif data_type == 'test':
            self.orig_data = x_test
            self.orig_y = y_test

        self.num_subjects = num_subjects
        self.thetas = []
        rng = np.random.default_rng(42)
        data_subj_ls, Ws = [], []
        for _ in range(self.num_subjects):
            theta = rng.uniform(-2*np.pi, 2*np.pi, 1)[0]
            self.thetas.append(theta)
            W_subj = np.array([[np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])

            data_subj = self.orig_data @ W_subj
            data_subj -= data_subj.mean(0)
            Ws.append(W_subj)
            data_subj_ls.append(data_subj)

        self.Ws = np.stack(Ws, axis=0)
        self.data_np = np.stack(data_subj_ls, axis=1)
        self.thetas = np.asarray(self.thetas)

        self.num_timesteps = self.data_np.shape[0]
        self.data = torch.from_numpy(self.data_np).float().view(-1, self.data_size)
        self.subject_ixs = torch.arange(self.num_subjects).unsqueeze(0).repeat(self.num_timesteps, 1).view(-1)
        self.temporal_ixs = torch.arange(self.num_timesteps).unsqueeze(1).repeat(1, self.num_subjects).view(-1)
        
        self.y = torch.from_numpy(self.orig_y).float().squeeze().unsqueeze(1).repeat(1, self.num_subjects).view(-1)

    def __len__(self):
        return self.num_subjects * self.num_timesteps

    @property
    def num_classes(self):
        return 1
    
    @property
    def data_size(self):
        return 2

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs[ix]
        return x, subject_ix, temporal_ix, y
