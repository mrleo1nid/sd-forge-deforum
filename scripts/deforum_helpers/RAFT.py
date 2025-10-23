# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small
)


class RAFT:
    """RAFT optical flow model wrapper.

    Supports two model sizes:
    - 'large': Best quality, slower (default)
    - 'small': Faster, good quality

    Args:
        model_size: 'large' or 'small'
        num_flow_updates: Number of flow refinement iterations (default: 12)
    """

    def __init__(self, model_size: str = 'large', num_flow_updates: int = 12):
        self.model_size = model_size.lower()
        self.num_flow_updates = num_flow_updates

        # Select model and weights based on size
        if self.model_size == 'small':
            weights = Raft_Small_Weights.DEFAULT
            model_fn = raft_small
        else:  # default to 'large'
            weights = Raft_Large_Weights.DEFAULT
            model_fn = raft_large

        self.transforms = weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading RAFT {self.model_size} model with {num_flow_updates} flow iterations...")
        self.model = model_fn(weights=weights, progress=False).to(self.device).eval()

    def predict(self, image1, image2, num_flow_updates: int = None):
        """Predict optical flow between two images.

        Args:
            image1: First image (PIL or numpy)
            image2: Second image (PIL or numpy)
            num_flow_updates: Override default flow iterations

        Returns:
            Flow array with shape (H, W, 2)
        """
        # Use instance default if not specified
        if num_flow_updates is None:
            num_flow_updates = self.num_flow_updates

        img1 = F.to_tensor(image1)
        img2 = F.to_tensor(image2)
        img1_batch, img2_batch = img1.unsqueeze(0), img2.unsqueeze(0)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)

        with torch.no_grad():
            flow = self.model(
                image1=img1_batch.to(self.device),
                image2=img2_batch.to(self.device),
                num_flow_updates=num_flow_updates
            )[-1].cpu().numpy()[0]

        # align the flow array to have the shape (H, W, 2) so it's compatible with the rest of CV2's flow methods
        flow = np.transpose(flow, (1, 2, 0))

        return flow

    def delete_model(self):
        """Delete model to free VRAM."""
        del self.model
