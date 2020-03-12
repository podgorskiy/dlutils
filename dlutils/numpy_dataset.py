# Copyright 2018-2020 Stanislav Pidhorskyi
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

import numpy as np


class NumpyDataset:
    @staticmethod
    def list_of_pairs_to_numpy(l):
        return np.asarray([x[1] for x in l], np.uint8), np.asarray([x[0] for x in l], np.int)

    def __init__(self, data):
        self.x, self.y = NumpyDataset.list_of_pairs_to_numpy(data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.y[index.start:index.stop], self.x[index.start:index.stop]
        return self.y[index], self.x[index]

    def __len__(self):
        return len(self.y)

    def shuffle(self):
        permutation = np.random.permutation(self.y.shape[0])
        for x in [self.y, self.x]:
            np.take(x, permutation, axis=0, out=x)
