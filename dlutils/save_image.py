# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import math
import numpy as np
from PIL import Image


def make_grid(images, nrow=8, padding=2, NCWH=False):
    if isinstance(images, list):
        images = np.stack(images, axis=0)
    if len(images.shape) == 2:
        images = images[None, ...]
    if len(images.shape) == 3:
        if images.shape[0 if NCWH else 2] == 1:
            images = np.concatenate((images, images, images), 0 if NCWH else 2)
        images = images[None, ...]
    if len(images.shape) == 4 and images.shape[1 if NCWH else 3] == 1:
        images = np.concatenate((images, images, images), 1 if NCWH else 3)
    if images.shape[0] == 1:
        return images[0]

    nmaps = images.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(images.shape[2 if NCWH else 1] + padding), int(images.shape[3 if NCWH else 2] + padding)
    num_channels = images.shape[1 if NCWH else 3]
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, num_channels), dtype=images.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if NCWH:
                grid[:, y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width] = images[k]
            else:
                grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width] = images[k]
            k = k + 1
    return grid


def save_image(images, filename, nrow=8, padding=2, NCWH=False, format=None):
    images = make_grid(images, nrow, padding, NCWH)
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = np.clip(images * 255 + 0.5, 0, 255).astype(np.uint8)
    if NCWH:
        images = images.transpose((1, 2, 0))
    im = Image.fromarray(images)
    im.save(filename, format=format)
