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

import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import yacs.config
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

import os


__all__ = ['LossTracker']


class RunningMean:
    def __init__(self):
        self._mean = 0.0
        self.n = 0

    def __iadd__(self, value):
        self._mean = (float(value) + self._mean * self.n)/(self.n + 1)
        self.n += 1
        return self

    def reset(self):
        self._mean = 0.0
        self.n = 0

    def mean(self):
        return self._mean


class RunningMeanTorch:
    def __init__(self):
        self.values = []

    def __iadd__(self, value):
        with torch.no_grad():
            self.values.append(value.cpu().unsqueeze(0))
            return self

    def reset(self):
        self.values = []

    def mean(self):
        with torch.no_grad():
            if len(self.values) == 0:
                return 0.0
            return float(torch.cat(self.values).mean().item())


def isinstance(o, c):
    return o.__class__.__name__ == c.__name__


class LossTracker:
    def __init__(self, output_dir='.'):
        self.tracks = OrderedDict()
        self.epochs = []
        self.means_over_epochs = OrderedDict()
        self.output_dir = output_dir.OUTPUT_DIR if isinstance(output_dir, yacs.config.CfgNode) else output_dir

    def update(self, d):
        for k, v in d.items():
            if k not in self.tracks:
                self.add(k, isinstance(v, torch.Tensor))
            self.tracks[k] += v

    def add(self, name, pytorch=True):
        assert name not in self.tracks, "Name is already used"
        if pytorch and has_torch:
            track = RunningMeanTorch()
        else:
            track = RunningMean()
        self.tracks[name] = track
        self.means_over_epochs[name] = []
        return track

    def register_means(self, epoch):
        self.epochs.append(epoch)

        for key in self.means_over_epochs.keys():
            if key in self.tracks:
                value = self.tracks[key]
                self.means_over_epochs[key].append(value.mean())
                value.reset()
            else:
                self.means_over_epochs[key].append(None)

        with open(os.path.join(self.output_dir, 'log.csv'), mode='w') as csv_file:
            fieldnames = ['epoch'] + list(self.tracks.keys())
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for i in range(len(self.epochs)):
                try:
                    writer.writerow([self.epochs[i]] + [self.means_over_epochs[x][-(len(self.epochs) - i)] for x in self.tracks.keys()])
                except IndexError:
                    pass

    def __str__(self):
        result = ""
        for key, value in self.tracks.items():
            result += "%s: %.7f, " % (key, value.mean())
        return result[:-2]

    def plot(self):
        plt.figure(figsize=(12, 8))
        for key in self.tracks.keys():
            plt.plot(self.epochs, self.means_over_epochs[key], label=key)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'plot.png'))
        plt.close()

    def state_dict(self):
        tracks = {}
        for key, track in self.tracks.items():
            t = {}
            if isinstance(track, RunningMean):
                t['type'] = RunningMean.__name__
                t['_mean'] = track._mean
                t['n'] = track.n
            elif isinstance(track, RunningMeanTorch):
                t['type'] = RunningMeanTorch.__name__
                t['values'] = track.values
            else:
                raise ValueError
            tracks[key] = t
        return {
            'tracks': tracks,
            'epochs': self.epochs,
            'means_over_epochs': self.means_over_epochs,
        }

    def load_state_dict(self, state_dict):
        self.epochs = state_dict['epochs']
        self.means_over_epochs = state_dict['means_over_epochs']

        tracks = state_dict['tracks']
        self.tracks = {}
        for key, track in tracks.items():
            if isinstance(track, RunningMean) or isinstance(track, RunningMeanTorch):
                self.tracks[key] = track
            else:
                if track['type'] == RunningMean.__name__:
                    rm = RunningMean()
                    rm._mean = track['_mean']
                    rm.n = track['n']
                    self.tracks[key] = rm
                elif track['type'] == RunningMeanTorch.__name__:
                    rm = RunningMeanTorch()
                    rm.values = track['values']
                    self.tracks[key] = rm
                else:
                    raise ValueError

        counts = list(map(len, self.means_over_epochs.values()))

        if len(counts) == 0:
            counts = [0]
        m = min(counts)

        if m < len(self.epochs):
            self.epochs = self.epochs[:m]

        for key in self.means_over_epochs.keys():
            if len(self.means_over_epochs[key]) > m:
                self.means_over_epochs[key] = self.means_over_epochs[key][:m]
