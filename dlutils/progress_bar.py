# Copyright 2018-2019 Stanislav Pidhorskyi
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

import sys
import os
import time


class ProgressBar:
    def __init__(self, total_iterations, prefix='Progress:', suffix='', decimals=1, length=None, fill='#'):
        self.format_string = "\r%s |%%s| %%.%df%%%% [%%d/%d] %s" % (prefix, decimals, total_iterations, suffix)
        self.total_iterations = total_iterations
        self.length = length
        self.fill = fill
        self.current_iteration = 0
        self.file = sys.stderr
        self.last_print = -1
        self.min_print_interval = 0.3
        try:
            self.columns = os.get_terminal_size().columns
        except (AttributeError, OSError):
            self.columns = 80
        if self.length is None:
            self.length = self.columns - len(self._get_status_string('', 100)) - 1

    def _get_status_string(self, bar, percent):
        return self.format_string % (bar, percent, self.current_iteration)

    def increment(self, val=1):
        self.current_iteration += val
        current_time = time.time()
        delta_time = current_time - self.last_print

        if delta_time >= self.min_print_interval or self.current_iteration == self.total_iterations:
            self.last_print = current_time
            percent = 100 * (self.current_iteration / float(self.total_iterations))
            filled_length = int(self.length * self.current_iteration // self.total_iterations)
            bar = self.fill * filled_length + '-' * (self.length - filled_length)
            self.file.write(self._get_status_string(bar, percent))
            self.file.flush()
            if self.current_iteration == self.total_iterations:
                self.file.write('\n')

if __name__ == '__main__':
    items = range(1000)
    l = len(items)

    pb = ProgressBar(l)

    for item in items:
        time.sleep(0.01)
        pb.increment()
