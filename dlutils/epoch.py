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
import time


class EpochRange:
    """ Range for iterating epochs """
    def __init__(self, epoch_count, log_func=None):
        self._tracker = LossTracker()
        self._epochs = range(epoch_count).__iter__()
        self._epoch_count = epoch_count
        self._epoch_start_time = 0
        self._current_epoch = -1
        self._format = '[%d/%d] - time: %.2f'
        self._log_func = log_func if log_func is not None else print

    def __len__(self):
        return len(self._epoch_count)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_epoch != -1:
            per_epoch_time = time.time() - self._epoch_start_time
            epoch_status_str = self._format % ((self._current_epoch + 1), self._epoch_count, per_epoch_time)
            if len(self._tracker.tracks) > 0:
                epoch_status_str = '; '.join([epoch_status_str, str(self._tracker)])
            self._log_func(epoch_status_str)

        self._epoch_start_time = time.time()
        self._tracker = LossTracker()
        self._current_epoch = self._epochs.__next__()
        return self._current_epoch, self._tracker


class RunningMean:
    def __init__(self):
        self.mean = 0.0
        self.n = 0

    def __iadd__(self, value):
        self.mean = (float(value) + self.mean * self.n)/(self.n + 1)
        self.n += 1
        return self

    def __float__(self):
        return self.mean


class LossTracker:
    """ Tracker for easy recording and computing mean values of some quanities such as losses. Summary of average values is printed at the end of each epoch.
    """
    def __init__(self):
        self.tracks = {}

    def add(self, name, format_str="%s: %.3f"):
        track = RunningMean()
        self.tracks[name] = (track, format_str)
        return track

    def reset(self):
        for name, (track, format_str) in self.tracks.items():
            track.mean = 0.0
            track.n = 0

    def __str__(self):
        return ', '.join([format_str % (name, track.mean) for name, (track, format_str) in self.tracks.items()])
