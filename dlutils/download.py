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
"""Module for downloading files, downloading files from google drive, uncompressing targz"""

from __future__ import print_function
import os
import cgi
import tarfile
import gzip
import shutil
import zipfile
try:
    from urllib import request
    from http import cookies, cookiejar
except ImportError:
    # Fall back to Python 2
    import urllib2 as request
    import Cookie as cookies
    import cookielib as cookiejar


def from_google_drive(google_drive_fileid, directory=".", file_name=None, extract_targz=False, extract_gz=False, extract_zip=False):
    """ Downloads file from Google Drive.

    Given the file ID, file is downloaded from Google Drive and optionally it can be unpacked after downloading
    completes.

    Note:
        You need to share the file as ``Anyone who has the link can access. No sign-in required.``. You can find the
        file ID in the link:

        `https://drive.google.com/file/d/` ``0B3kP5zWXwFm_OUpQbDFqY2dXNGs`` `/view?usp=sharing`

    Args:
        google_drive_fileid (str): file ID.
        directory (str): Directory where to save the file
        file_name (str, optional): If not None, this will overwrite the file name, otherwise it will use the filename
            returned from http request. Defaults to None.
        extract_targz (bool): Extract tar.gz archive. Defaults to False.
        extract_gz (bool): Decompress gz compressed file. Defaults to False.
        extract_zip (bool): Extract zip archive. Defaults to False.

    Example:

        ::

            dlutils.download.from_google_drive(directory="data/", google_drive_fileid="0B3kP5zWXwFm_OUpQbDFqY2dXNGs")

    """
    url = "https://drive.google.com/uc?export=download&id=" + google_drive_fileid
    cj = cookiejar.CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(cj))
    u = opener.open(url)
    cookie = cookies.SimpleCookie()
    c = u.info().get("set-cookie")
    if c:
        cookie.load(c)
    token = ""
    for key, value in cookie.items():
        if key.startswith('download_warning'):
            token = value.value
    if c:
        url += "&confirm=" + token
    request_obj = opener.open(url)
    _download(request_obj, url, directory, file_name, extract_targz, extract_gz, extract_zip)


def from_url(url, directory=".", file_name=None, extract_targz=False, extract_gz=False, extract_zip=False):
    """ Downloads file from specified URL.

    Optionally it can be unpacked after downloading completes.

    Args:
        url (str): file URL.
        directory (str): Directory where to save the file
        file_name (str, optional): If not None, this will overwrite the file name, otherwise it will use the filename
            returned from http request. Defaults to None.
        extract_targz (bool): Extract tar.gz archive. Defaults to False.
        extract_gz (bool): Decompress gz compressed file. Defaults to False.
        extract_zip (bool): Extract zip archive. Defaults to False.

    Example:

        ::

            dlutils.download.from_url("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", directory, extract_gz=True)

    """
    request_obj = request.urlopen(url)
    _download(request_obj, url, directory, file_name, extract_targz, extract_gz, extract_zip)


def _download(request_obj, url, directory, file_name, extract_targz, extract_gz, extract_zip):
    meta = request_obj.info()

    if file_name is None:
        cd = meta.get("content-disposition")
        if cd is not None:  
            value, params = cgi.parse_header(cd)
            cd_file = params['filename']
            if cd_file is not None:
                file_name = cd_file

    if file_name is None:
        file_name = url.split('/')[-1]

    file_path = os.path.join(directory, file_name)

    file_size = 0
    length_header = meta.get("Content-Length")
    if length_header is not None:
        file_size = int(length_header)
        print("Downloading: %s Bytes: %d" % (file_name, file_size))
    else:
        print("Downloading: %s" % file_name)

    if os.path.exists(file_path) and (os.path.getsize(file_path) == file_size or file_size == 0):
        print("File %s already exists, skipping" % file_path)
        return

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as file:
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = request_obj.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            file.write(buffer)
            if file_size > 0:
                status = "\r%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            else:
                status = "\r%10d" % file_size_dl
            print(status, end='')

        print()

    if extract_targz:
        print("Extracting...")
        tarfile.open(name=file_path, mode="r:gz").extractall(directory)

    if extract_gz:
        file_out_path = file_path.replace('.gz', '')

        print("Extracting...")
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if extract_zip:
        print("Extracting...")
        zipfile.ZipFile(file_path, 'r').extractall(directory)
    print("Done")


def mnist(directory='mnist'):
    """Downloads `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        directory (str): Directory where to save the files

    """
    from_url("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", directory, extract_gz=True)
    from_url("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", directory, extract_gz=True)
    from_url("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", directory, extract_gz=True)
    from_url("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", directory, extract_gz=True)


def fashion_mnist(directory='fashion-mnist'):
    """Downloads `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        directory (str): Directory where to save the files

    """
    from_url("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", directory, extract_gz=True)
    from_url("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", directory, extract_gz=True)
    from_url("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", directory, extract_gz=True)
    from_url("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", directory, extract_gz=True)


def cifar10(directory='cifar10'):
    """Downloads `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        directory (str): Directory where to save the files

    """
    from_url("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", directory, extract_targz=True)


def cifar100(directory='cifar100'):
    """Downloads `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        directory (str): Directory where to save the files

    """
    from_url("https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz", directory, extract_targz=True)
