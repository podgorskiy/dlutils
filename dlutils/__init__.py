from .batch_provider import batch_provider
from . import mnist_reader
from . import cifar10_reader
from . import cifar100_reader
from . import downloader


def download_mnist(directory='mnist'):
    downloader.from_url("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", directory, extract_gz=True)
    downloader.from_url("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", directory, extract_gz=True)
    downloader.from_url("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", directory, extract_gz=True)
    downloader.from_url("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", directory, extract_gz=True)

