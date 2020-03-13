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

import threading

__all__ = ['async_func']


class AsyncCall(object):
    def __init__(self, fnc, callback=None):
        self.callable = fnc
        self.callback = callback
        self.result = None

    def __call__(self, *args, **kwargs):
        self.thread = threading.Thread(target=self.run, name=self.callable.__name__, args=args, kwargs=kwargs)
        self.thread.start()
        return self

    def wait(self, timeout=None):
        self.thread.join(timeout)
        if self.thread.isAlive():
            raise TimeoutError
        else:
            return self.result

    def run(self, *args, **kwargs):
        self.result = self.callable(*args, **kwargs)
        if self.callback:
            self.callback(self.result)


class AsyncMethod(object):
    def __init__(self, fnc, callback=None):
        self.callable = fnc
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return AsyncCall(self.callable, self.callback)(*args, **kwargs)


def async_func(fnc=None, callback=None):
    if fnc is None:
        def add_async_callback(f):
            return AsyncMethod(f, callback)
        return add_async_callback
    else:
        return AsyncMethod(fnc, callback)
