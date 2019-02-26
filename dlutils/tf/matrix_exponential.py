# Copyright 2017 Stanislav Pidhorskyi
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
"""Matrix exponential for tensorflow"""

import tensorflow as tf


def matrix_exponential(x):
    r_tr2 = tf.matmul(x, x)
    r_tr3 = tf.matmul(r_tr2, x)
    r_tr4 = tf.matmul(r_tr3, x)
    r_tr5 = tf.matmul(r_tr4, x)
    r_tr6 = tf.matmul(r_tr5, x)
    r_tr7 = tf.matmul(r_tr6, x)
    r_tr8 = tf.matmul(r_tr7, x)
    r_tr9 = tf.matmul(r_tr8, x)
    r_tr10 = tf.matmul(r_tr9, x)
    r_tr11 = tf.matmul(r_tr10, x)
    r_tr12 = tf.matmul(r_tr11, x)
    r_tr13 = tf.matmul(r_tr12, x)
    result = tf.eye(x.get_shape(1))\
             + x \
             + 1.0 / 2.0 * r_tr2\
             + 1.0 / 6.0 * r_tr3\
             + 1.0 / 24.0 * r_tr4\
             + 1.0 / 120.0 * r_tr5\
             + 1.0 / 720.0 * r_tr6\
             + 1.0 / 5040.0 * r_tr7\
             + 1.0 / 40320.0 * r_tr8\
             + 1.0 / 362880.0 * r_tr9\
             + 1.0 / 3628800.0 * r_tr10\
             + 1.0 / 39916800.0 * r_tr11\
             + 1.0 / 479001600.0 * r_tr12\
             + 1.0 / 6227020800.0 * r_tr13
    return result
