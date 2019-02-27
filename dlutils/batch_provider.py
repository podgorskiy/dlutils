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

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
from threading import Thread, Lock, Event
from .progress_bar import ProgressBar


def batch_provider(data, batch_size, processor, worker_count=1, queue_size=16, report_progress=False):
    """
    Returns iterable object that iterates over a list of data.
    Custom processing function can be applied to each batch.
    Processes batches in parallel, and asynchronously fills a queue of next batches.
    """

    class State:
        def __init__(self):
            self.current_batch = 0
            self.lock = Lock()
            self.data_len = len(data)
            self.batch_count = self.data_len // batch_size + (1 if self.data_len % batch_size != 0 else 0)
            self.quit_event = Event()
            self.queue = Queue(queue_size)
            self.batches_done_count = 0
            self.progress_bar = None
            if report_progress:
                self.progress_bar = ProgressBar(self.batch_count)

        def get_next_batch_it(self):
            try:
                self.lock.acquire()
                if self.quit_event.is_set() or self.current_batch == self.batch_count:
                    raise StopIteration
                cb = self.current_batch
                self.current_batch += 1
                return cb
            finally:
                self.lock.release()

        def push_done_batch(self, batch):
            try:
                self.lock.acquire()
                state.queue.put(batch)
                self.batches_done_count += 1
            finally:
                self.lock.release()

        def all_done(self):
            return self.batches_done_count == self.batch_count and state.queue.empty()

    state = State()

    def _worker():
        while not state.quit_event.is_set():
            try:
                cb = state.get_next_batch_it()
                data_slice = data[cb * batch_size:min((cb + 1) * batch_size, state.data_len)]
                b = processor(data_slice)
                state.push_done_batch(b)
            except StopIteration:
                break

    def _generator():
        workers = []
        for i in range(worker_count):
            worker = Thread(target=_worker)
            worker.daemon = True
            worker.start()
            workers.append(worker)
        try:
            while not state.quit_event.is_set() and not state.all_done():
                item = state.queue.get()
                state.queue.task_done()
                yield item
                if state.progress_bar is not None:
                    state.progress_bar.increment()

        except GeneratorExit:
            state.quit_event.set()
            while not state.queue.empty():
                try:
                    state.queue.get(False)
                except Empty:
                    continue
            state.queue.task_done()

    class Iterator:
        def __init__(self, batch_count, generator):
            self.batch_count = batch_count
            self.generator = generator

        def __len__(self):
            return self.batch_count

        def __iter__(self):
            return self.generator

        def __next__(self):
            return self.generator.next()

        #def __del__(self):
        #    print("Exiting")

    return Iterator(state.batch_count, _generator())
