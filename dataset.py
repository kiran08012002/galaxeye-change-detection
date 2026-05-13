import numpy as np
from torch.utils.data import Dataset


class EOSARChangeDataset(Dataset):
    """
    EO-SAR Change Detection Dataset.

    Each scene consists of three images stored sequentially in the HuggingFace
    dataset split:
        - Pre-event  (EO,  RGB  3-channel) at index pre_start  + scene_idx
        - Post-event (SAR, gray 1-channel) at index post_start + scene_idx
        - Target mask (binary 0/1)         at index target_start + scene_idx

    The four modality channels [R, G, B, SAR] are concatenated and normalised
    to [0, 1] before being passed to the model.
    """

    def __init__(self, hf_dataset, pre_start, post_start,
                 target_start, n_scenes, transform=None):
        self.ds           = hf_dataset
        self.pre_start    = pre_start
        self.post_start   = post_start
        self.target_start = target_start
        self.n_scenes     = n_scenes
        self.transform    = transform

    def __len__(self):
        return self.n_scenes

    def __getitem__(self, idx):
        pre  = np.array(self.ds[self.pre_start  + idx]['image'])  # (H,W,3)
        post = np.array(self.ds[self.post_start + idx]['image'])  # (H,W)
        mask = np.array(self.ds[self.target_start + idx]['image']).astype(np.float32)

        # Ensure pre is (H,W,3)
        if pre.ndim == 2:
            pre = np.stack([pre, pre, pre], axis=-1)
        elif pre.shape[2] == 4:
            pre = pre[:, :, :3]

        # Ensure post is (H,W,1)
        if post.ndim == 3:
            post = post[:, :, 0:1]
        else:
            post = post[:, :, np.newaxis]

        # Concatenate → (H,W,4) normalised to [0,1]
        image_4ch = np.concatenate([pre, post], axis=-1).astype(np.float32) / 255.0

        if self.transform:
            aug       = self.transform(image=image_4ch, mask=mask)
            image_4ch = aug['image']
            mask      = aug['mask']

        return image_4ch, mask
