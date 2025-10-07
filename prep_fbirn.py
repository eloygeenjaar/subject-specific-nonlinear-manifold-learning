import nilearn
import nibabel as nb
import numpy as np
from pathlib import Path
from nilearn import signal, masking, image
from sklearn.model_selection import train_test_split

data_p = Path('/data/qneuromark/Data/FBIRN/ZN_Neuromark/ZN_Prep_fMRI/')
subjects = list(data_p.iterdir())

mask_img = nb.load('mask_ICV.nii')
img = nb.load(subjects[0] / 'SM.nii')
mask_img = image.resample_to_img(mask_img, img, interpolation='nearest')
mask_data = mask_img.get_fdata().astype(int)


avg_ls = []
data_ls = []
subjects_ls = []
fmri_imgs = []
s_ix = 0
for subject in subjects:
    func_p = subject / 'SM.nii'
    if func_p.is_file():
        fmri_imgs.append(nb.load(func_p))

epi_mask_img = masking.compute_multi_epi_mask(fmri_imgs, n_jobs=-1, verbose=1, opening=2, threshold=0.5)
epi_mask_data = epi_mask_img.get_fdata().astype(int)
mask_data = mask_data & epi_mask_data

mask_img = nb.Nifti1Image(mask_data.astype(bool), affine=mask_img.affine, header=mask_img.header)
nb.save(mask_img, 'images/fbirn_mask.nii')

#mask_data = mask_data.reshape(-1)
for subject in subjects:
    func_p = subject / 'SM.nii'
    if func_p.is_file():
        fmri = nb.load(func_p)
        fmri = masking.apply_mask([fmri], mask_img)
        assert fmri.shape[0] == 157
        assert fmri.shape[-1] == mask_data.sum()
        print(fmri.shape)
        fmri = signal.clean(fmri, t_r=2.0, high_pass=None, detrend=True, standardize='zscore_sample')
        data_ls.append(fmri.astype(np.float16))
        subjects_ls.append(subject)
        s_ix += 1
        print(s_ix)

data = np.stack(data_ls, axis=0).astype(np.float16)
num_subjects = data.shape[0]
subjects = np.asarray(subjects_ls)

train_subjects = int(0.8 * num_subjects)

ix_tr = np.arange(157)

# Take first half for training + validation and second half for test
ix_train, ix_valid = train_test_split(
    ix_tr, random_state=42, shuffle=False, train_size=0.75)
print(ix_train, ix_valid)

np.savez_compressed('/data/users1/egeenjaar/data/fbirn_temp/train.npz',
                    data=data[:train_subjects, ix_train],
                    subjects=subjects[:train_subjects])
np.savez_compressed('/data/users1/egeenjaar/data/fbirn_temp/valid.npz',
                    data=data[:train_subjects, ix_valid],
                    subjects=subjects[:train_subjects])
np.savez_compressed('/data/users1/egeenjaar/data/fbirn_temp/test.npz',
                    data=data[train_subjects:, :],
                    subjects=subjects[train_subjects:])
