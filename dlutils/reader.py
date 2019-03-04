# Copyright 2017-2019 Stanislav Pidhorskyi
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
"""Util for reading MNIST dataset"""

import mmap
import os
import numpy as np
from contextlib import closing
from scipy import misc


class Mnist:
    """Read MNIST out of binary batches"""
    def __init__(self, path, items=None, train=True, test=False, resize_to_32x32=False):
        self.items = []

        self._path = path
        self._label_bytes = 1
        height = 28
        width = 28
        self._image_bytes = height * width
        self._record_bytes = self._image_bytes  # stride of items in bin file
        self._resize_to_32x32 = resize_to_32x32
        
        if items is not None:
            self.items = items
        else:
            if train:
                self.__read_batch('train-labels-idx1-ubyte', 'train-images-idx3-ubyte', 60000)

            if test:
                self.__read_batch('t10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 10000)

    def __read_batch(self, batch_label, batch_images, n):
        """Read MNIST binary batch using mmap"""
        with open(os.path.join(self._path, batch_label), 'rb') as f_l:
            with open(os.path.join(self._path, batch_images), 'rb') as f_i:
                with closing(mmap.mmap(f_l.fileno(), length=0, access=mmap.ACCESS_READ)) as m_l:
                    with closing(mmap.mmap(f_i.fileno(), length=0, access=mmap.ACCESS_READ)) as m_i:
                        for i in range(n):
                            l = m_l[i + 8]
                            try:
                                # Python 2
                                label = ord(l)
                            except TypeError:
                                # Python 3
                                label = l
                            img = np.fromstring(
                                m_i[16 + i * self._record_bytes
                                    :16 + i * self._record_bytes + self._record_bytes], dtype=np.uint8)
                            img = np.reshape(img, (28, 28))
                            if self._resize_to_32x32:
                                img = misc.imresize(img, (32, 32), interp='bilinear')
                            self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]


class Cifar10:
    """Read CIFAR out of binary batches"""
    def __init__(self, path, train=True, test=False):
        self.items = []

        self._path = path
        self._label_bytes = 1
        height = 32
        width = 32
        depth = 3
        self._image_bytes = height * width * depth
        self._record_bytes = self._label_bytes + self._image_bytes  # stride of items in bin file
        self._item_count = 10000

        if train:
            self.__read_batch('data_batch_1.bin')
            self.__read_batch('data_batch_2.bin')
            self.__read_batch('data_batch_3.bin')
            self.__read_batch('data_batch_4.bin')
            self.__read_batch('data_batch_5.bin')

        if test:
            self.__read_batch('test_batch.bin')

    def __read_batch(self, batch):
        """Read CIFAR binary batch using mmap"""
        with open(os.path.join(self._path, batch), 'rb') as f:
            with closing(mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)) as m:
                for i in range(self._item_count):
                    l = m[i * self._record_bytes]
                    try:
                        # Python 2
                        label = ord(l)
                    except TypeError:
                        # Python 3
                        label = l
                    img = np.fromstring(
                        m[i * self._record_bytes + self._label_bytes
                          :i * self._record_bytes + self._record_bytes], dtype=np.uint8)
                    img = np.reshape(img, (3, 32, 32))
                    self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]


class Cifar100:
    """Read CIFAR out of binary batches"""
    def __init__(self, path, train=True, test=False):
        self.items = []

        self._path = path
        self._label_bytes = 2
        height = 32
        width = 32
        depth = 3
        self._image_bytes = height * width * depth
        self._record_bytes = self._label_bytes + self._image_bytes # stride of items in bin file

        if train:
            self._read_batch('train.bin', 50000)

        if test:
            self._read_batch('test.bin', 10000)

    def _read_batch(self, batch, n):
        """Read CIFAR binary batch using mmap"""
        with open(os.path.join(self._path, batch), 'rb') as f:
            with closing(mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)) as m:
                for i in range(n):
                    l = m[i * self._record_bytes] + m[i * self._record_bytes + 1] * 0x100
                    try:
                        # Python 2
                        label = ord(l)
                    except TypeError:
                        # Python 3
                        label = l
                    img = np.fromstring(
                        m[i * self._record_bytes + self._label_bytes
                          :i * self._record_bytes + self._record_bytes], dtype=np.uint8)
                    img = np.reshape(img, (3, 32, 32))
                    self.items.append((label, img))

    def get_labels(self):
        return [item[0] for item in self.items]

    def get_images(self):
        return [item[1] for item in self.items]
