# Copyright 2020 Stanislav Pidhorskyi
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


def block_process_2d(data, func, block_size=512, overlap=32, intermediate_as_double=False):
    """ Applies function to the given 4-dimensional tensor by chucking it in 2 dimensions and processing each chunk
    separately.

    Input data is split into chuck across last two dimension. If input data is represented as NCHW, chucking will occur
    in HW space. Size of chunks and overlap can be modified. Overlap tells how much chunks should overlap.

    Overlapping regions will be interpolated linearly/bilinearly between chunks.

    Note:
        Type of data in tensor `data` is expected to be float-point one (either float or double)

    Args:
        data (torch.Tensor): Input data to be processed by `func`.
        func (Callable): Function to be used on `data`
        block_size (int): Size of chunks
        overlap (int): Tells how much chunks should overlap
        intermediate_as_double (bool): Use `double` for intermidiate representation, improves accuracy

    Returns:
        torch.Tensor or list[torch.Tensor]: Result of the function `func` applied to input `data`

    Example:

        ::

            def f(x):
                assert x.shape[2] <= 64
                assert x.shape[3] <= 64
                return x * x + x * x

            x = torch.ones(3, 3, 512, 512, dtype=torch.float32)
            r = dlutils.block_process_2d(x, f, block_size=32, overlap=8)

    """

    if not 0.0 != len(data.shape):
        raise ValueError("Invalid dimensionality of input data: {}".format(data.shape))
    if 0 > overlap:
        raise ValueError("Invalid value for overlap: {}".format(overlap))

    width = data.shape[-1]
    height = data.shape[-2]

    blocks = []

    for i in range((width + block_size - overlap - 1) // (block_size - overlap)):
        offset_x = i * (block_size - overlap)
        offset_x = min(offset_x + block_size, width) - block_size
        w = min(offset_x + block_size, width) - offset_x
        for j in range((height + block_size - overlap - 1) // (block_size - overlap)):
            offset_y = j * (block_size - overlap)
            offset_y = min(offset_y + block_size, height) - block_size
            h = min(offset_y + block_size, height) - offset_y

            blocks.append((offset_x, offset_y, w, h))

    results = []

    for offset_x, offset_y, w, h in blocks:
        res = func(data[:, :, offset_y:offset_y + h, offset_x:offset_x + w])
        if isinstance(res, list):
            if not all(isinstance(x, torch.Tensor) for x in res):
                raise ValueError("Function must returs either torch.Tensor, either list of torch.Tensor. But got list of {}".format([type(x) for x in res]))
        elif isinstance(res, torch.Tensor):
            pass
        else:
            raise ValueError("Function must return either torch.Tensor, either list of torch.Tensor. But got list of {}".format(type(res)))
        results.append(res)

    output = []

    returns_value = False

    if all(isinstance(x, list) for x in results):
        pass
    elif all(isinstance(x, torch.Tensor) for x in results):
        _results = [[x] for x in results]
        results = _results
        returns_value =True
    else:
        raise ValueError("Function must return either torch.Tensor, either list of torch.Tensor.")

    for tensor in results[0]:
        output.append(torch.zeros(*tensor.shape[:2], height, width, dtype=torch.double if intermediate_as_double else tensor.dtype))

    counts = torch.zeros(1, 1, height, width, dtype=torch.double)

    weight_mask = torch.ones(1, 1, block_size, block_size, dtype=torch.double)

    for i in range(overlap):
        weight_mask[:, :, :, i] *= ((i + 1) / overlap)
        weight_mask[:, :, :, -i] *= ((i + 1) / overlap)

    for i in range(overlap):
        weight_mask[:, :, i, :] *= ((i + 1) / overlap)
        weight_mask[:, :, -i, :] *= ((i + 1) / overlap)

    for block, res in zip(blocks, results):
        offset_x, offset_y, w, h = block
        counts[:, :, offset_y:offset_y + h, offset_x:offset_x + w] += weight_mask
        for o, r in zip(output, res):
            o[:, :, offset_y:offset_y + h, offset_x:offset_x + w] += r * weight_mask

    if intermediate_as_double:
        _output = []
        for o, i in zip(output, results[0]):
            o /= counts
            _output.append(o.type(i.dtype))
        output = _output
        del _output
    else:
        for o in output:
            o /= counts

    if returns_value:
        return output[0]
    return output


if __name__ == '__main__':
    def f(x):
        assert x.shape[2] <= 64
        assert x.shape[3] <= 64
        return x * x + x * x

    def f2(x):
        assert x.shape[2] <= 64
        assert x.shape[3] <= 64
        return [x * 3, x * x + x * x]

    x = torch.ones(3, 3, 512, 512, dtype=torch.float32)
    r = block_process_2d(x, f, block_size=32, overlap=8)

    print(r)

    x = torch.ones(3, 3, 512, 512, dtype=torch.float32)
    r = block_process_2d(x, f2, block_size=32, overlap=8)

    print(r)

    x = torch.randn(3, 3, 512, 512, dtype=torch.float32)
    r = block_process_2d(x, f, block_size=32, overlap=8, intermediate_as_double=True)

    assert torch.all(torch.abs(r - (x * x + x * x)) < 1e-18)

    x = torch.randn(3, 3, 512, 512, dtype=torch.float32)
    r = block_process_2d(x, f, block_size=32, overlap=8, intermediate_as_double=False)

    assert torch.all(torch.abs(r - (x * x + x * x)) < 1e-5)

