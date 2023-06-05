#!/usr/bin/env python
# coding=utf-8
# @Time    : 2021/12/20 14:09
# @Author  : 江斌
# @Software: PyCharm

import os
from PyCameraList import __version__

from common import DIST_DIR, DIST_FILE_DICT


def main():
    os.chdir(DIST_DIR)
    wheel_name = DIST_FILE_DICT[__version__]
    cmd = f"twine upload {wheel_name}"
    os.system(cmd)


if __name__ == '__main__':
    main()
