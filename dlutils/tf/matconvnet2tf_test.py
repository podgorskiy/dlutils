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

import matconvnet2tf
from dlutils.download import from_url
import numpy as np
import scipy.ndimage
import tensorflow as tf


def main():
    """Test MatConvNet2TF"""

    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-128.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-1024.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat")
    from_url(url="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat")

    models = [
        'imagenet-vgg-f.mat'
        ,'imagenet-vgg-s.mat'
        ,'imagenet-vgg-m.mat'
        ,'imagenet-vgg-m-128.mat'
        ,'imagenet-vgg-m-1024.mat'
        ,'imagenet-vgg-m-2048.mat'
        ,'imagenet-vgg-verydeep-16.mat'
        ,'imagenet-vgg-verydeep-19.mat'
        ]

    image = np.array(scipy.ndimage.imread('alexnet224x224_input.jpg'), ndmin=4)

    for m in models:
        print("Model: " + m)
        with tf.Graph().as_default(), tf.Session() as session:
            model = matconvnet2tf.MatConvNet2TF(m, do_debug_print=True)
            session.run(tf.global_variables_initializer())
            result = model.net['prob'].eval(feed_dict={model.input: image}).reshape(-1)
            indices = np.flip(result.argsort(), 0)[:10]
            for i in indices:
                print(str(result[i] * 100.0) + "% " + model.net['classes'][i])

if __name__ == '__main__':
    main()
