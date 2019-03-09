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


def batch_provider(data, batch_size, processor=None, worker_count=1, queue_size=16, report_progress=True):
    """ Return an object that produces a sequence of batches from input data

    Input data is split into batches of size :attr:`batch_size` which are processed with function :attr:`processor`
    Data is split and processed by separate threads and dumped into a queue allowing continuous
    provision of data. The main purpose of this primitive is to provide easy to use tool for parallel batch
    processing/generation in background while main thread runs the main algorithm.
    Batches are processed in parallel, allowing better utilization of CPU cores and disk that may improve
    GPU utilization for DL tasks with Storage/IO bottleneck.

    This primitive can be used in various ways. For small datasets, the input :attr:`data` list may contain actual
    dataset, while :attr:`processor` function does from small to no data processing. For larger datasets, :attr:`data`
    list may contain just filenames or keys while :attr:`processor` function reads data from disk or db.
    
    There are many purposes that function :attr:`processor` can be used for, depending on your use case.

    - Reading data from disk or db
    - Data decoding, e.g. from JPEG.
    - Augmenting data, flipping, rotating adding nose, etc.
    - Concatenation of data, stacking to single ndarray, conversion to a tensor, uploading to GPU.
    - Data generation.
    
    Note:
        Sequential order of batches is guaranteed only if number of workers is 1 (Default), otherwise batches might
        be supplied out of order.

    Args:
        data (list): Input data, each entry in the list should be a separate data point.
        batch_size (int): Size of a batch. If size of data is not divisible by :attr:`batch_size`, then
            the last batch will have smaller size.
        processor (Callable[[list], Any], optional): Function for processing batches. Receives slice of the :attr:`data`
            list as input. Can return object of any type. Defaults to None.
        worker_count (int, optional): Number of workers, should be greater or equal to one. To process data in parallel
            and fully load CPU :attr:`worker_count` should be close to the number of CPU cores. Defaults to one.
        queue_size (int, optional): Maximum size of the queue, which is number of batches to buffer. Should be larger
            than :attr:`worker_count`. Typically, one would want this to be as large as possible to amortize all disk
            IO and computational costs. Downside of large value is increased RAM consumption. Defaults to 16.
        report_progress (bool, optional): Print a progress bar similar to `tqdm`. You still may use `tqdm` if you set
            :attr:`report_progress` to False. To use `tqdm` just do

            ::

                for x in tqdm(batch_provider(...)):
                    ...

            Defaults to True.

    Returns:
        Iterator: An object that produces a sequence of batches. :meth:`next()` method of the iterator will return
        object that was produced by :attr:`processor` function


    Raises:
        StopIteration: When all data was iterated through. Stops the for loop.

    Example:

        ::

            def process(batch):
                images = [misc.imread(x[0]) for x in batch]
                images = np.asarray(images, dtype=np.float32)
                images = images.transpose((0, 3, 1, 2))
                labeles = [x[1] for x in batch]
                labeles = np.asarray(labeles, np.int)
                return torch.from_numpy(images) / 255.0, torch.from_numpy(labeles)

            data = [('some_list.jpg', 1), ('of_filenames.jpg', 2), ('etc.jpg', 4), ...] # filenames and labels
            batches = dlutils.batch_provider(data, 32, process)

            for images, labeles in batches:
                result = model(images)
                loss = F.nll_loss(result, labeles)
                loss.backward()
                optimizer.step()


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
                self.queue.put(batch)
                self.batches_done_count += 1
            finally:
                self.lock.release()

        def all_done(self):
            return self.batches_done_count == self.batch_count and self.queue.empty()

    if processor is None:
        def processor(x):
            return x

    def _worker(state):
        while not state.quit_event.is_set():
            try:
                cb = state.get_next_batch_it()
                data_slice = data[cb * batch_size:min((cb + 1) * batch_size, state.data_len)]
                b = processor(data_slice)
                state.push_done_batch(b)
            except StopIteration:
                break

    class Iterator:
        def __init__(self):
            self.state = State()

            self.workers = []
            for i in range(worker_count):
                worker = Thread(target=_worker, args=(self.state, ))
                worker.daemon = True
                worker.start()
                self.workers.append(worker)

        def __len__(self):
            return self.state.batch_count

        def __iter__(self):
            return self

        def __next__(self):
            if not self.state.quit_event.is_set() and not self.state.all_done():
                item = self.state.queue.get()
                self.state.queue.task_done()
                if self.state.progress_bar is not None:
                    self.state.progress_bar.increment()
                return item
            else:
                self.state.quit_event.set()
                raise StopIteration

        def __del__(self):
            self.state.quit_event.set()
            while not self.state.queue.empty():
                self.state.queue.get(False)
                self.state.queue.task_done()
            for worker in self.workers:
                worker.join()

    return Iterator()
