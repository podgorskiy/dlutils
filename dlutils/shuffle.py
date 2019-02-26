# Copyright 2019 Stanislav Pidhorskyi
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


def shuffle_ndarray(x, axis=0):
    np.take(x, np.random.permutation(x.shape[axis]), axis=axis, out=x)


def shuffle_ndarrays_in_unison(arrays, axis=0):
    assert(all(x.shape == arrays[0].shape for x in arrays))

    permutation = np.random.permutation(arrays[0].shape[axis])

    for x in arrays:
        np.take(x, permutation, axis=axis, out=x)


if __name__ == '__main__':
    a = np.asarray([['a', 'b'], ['c', 'd'], ['e', 'f']])
    b = np.asarray([[1, 5], [0, 2], [0, 1]])

    print(a)
    print(b)

    shuffle_ndarrays_in_unison([a, b], axis=0)

    print(a)
    print(b)

