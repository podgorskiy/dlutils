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
"""Profiling utils"""

import time


def timer(f):
    """ Decorator for timeing function (method) execution time.

    After return from function will print string: ``func: <function name> took: <time in seconds> sec``.

    Args:
        f (Callable[Any]): function to decorate.

    Returns:
        Callable[Any]: Decorated function.

    Example:

        ::

            >>> from dlutils import timer
            >>> @timer.timer
            ... def foo(x):
            ...     for i in range(x):
            ...             pass
            ...
            >>> foo(100000)
            func:'foo'  took: 0.0019 sec

    """
    def __wrapper(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        print('func:%r  took: %2.4f sec' % (f.__name__, time_end - time_start))
        return result
    return __wrapper
