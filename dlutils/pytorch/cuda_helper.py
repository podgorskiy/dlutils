# Copyright 2018 Stanislav Pidhorskyi
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

import torch
import dlutils

if dlutils.use_cuda is None or dlutils.use_cuda:
    dlutils.use_cuda = torch.cuda.is_available()

if dlutils.use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    DoubleTensor = torch.cuda.DoubleTensor
    FloatTensor = torch.cuda.FloatTensor
    HalfTensor = torch.cuda.HalfTensor
    LongTensor = torch.cuda.LongTensor
    IntTensor = torch.cuda.IntTensor
    ShortTensor = torch.cuda.ShortTensor
    CharTensor = torch.cuda.CharTensor
    ByteTensor = torch.cuda.ByteTensor

    print("Running on ", torch.cuda.get_device_name(device))

    del device
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    DoubleTensor = torch.DoubleTensor
    FloatTensor = torch.FloatTensor
    HalfTensor = torch.HalfTensor
    LongTensor = torch.LongTensor
    IntTensor = torch.IntTensor
    ShortTensor = torch.ShortTensor
    CharTensor = torch.CharTensor
    ByteTensor = torch.ByteTensor

    print("Running on CPU")
