#!/usr/bin/env python
# coding=utf-8
# @Time    : 2021/12/20 14:10
# @Author  : 江斌
# @Software: PyCharm

import os
import sys
from common import DIST_DIR, DIST_FILE_DICT, PROJECT_DIR

sys.path.append(str(PROJECT_DIR))
from PyCameraList import __version__


def main():
    os.chdir(DIST_DIR)
    wheel_name = DIST_FILE_DICT[__version__]
    cmd = f"pip install --force-reinstall {wheel_name}"
    os.system(cmd)


if __name__ == '__main__':
    main()
