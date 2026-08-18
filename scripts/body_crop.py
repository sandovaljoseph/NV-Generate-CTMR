# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from scipy import ndimage


def crop_img_body_mask(synthetic_images, combine_label, a_min=-1000):
    """Set voxels outside the body mask to the modality background value."""
    synthetic_images[combine_label == 0] = a_min
    return synthetic_images


def fill_body_envelope_holes(combine_label, body_label=200):
    """Fill background cavities enclosed by each axial body silhouette."""
    if combine_label.ndim != 5 or combine_label.shape[1] != 1:
        raise ValueError(
            "Body-mask cropping requires labels shaped (batch, 1, x, y, z); "
            f"got {tuple(combine_label.shape)}"
        )
    body = combine_label.detach().cpu().numpy() != 0
    filled_labels = combine_label.clone()
    for index in range(body.shape[0]):
        filled = body[index, 0].copy()
        for axial_index in range(body.shape[-1]):
            filled[:, :, axial_index] = ndimage.binary_fill_holes(
                body[index, 0, :, :, axial_index]
            )
        holes = torch.as_tensor(filled & ~body[index, 0], device=combine_label.device)
        filled_labels[index, 0][holes] = body_label
    return filled_labels
