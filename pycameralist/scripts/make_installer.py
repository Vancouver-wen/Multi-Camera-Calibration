#!/usr/bin/env python
# coding=utf-8
# @Time    : 2021/12/20 14:09
# @Author  : 江斌
# @Software: PyCharm

import os
import shutil
from common import PROJECT_DIR, BUILD_DIR


class MakeInstaller(object):
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.build_dir = BUILD_DIR
        os.chdir(self.project_dir)

    def remove_build_dir(self):
        if os.path.isdir(self.build_dir):
            print(f'【remove_build_dir】 {self.build_dir}')
            shutil.rmtree(self.build_dir)
        else:
            print(f'【remove_build_dir】 目录不存在{self.build_dir}')

    def build_ext(self):
        cmd = "python setup.py build_ext --inplace"
        print(f"【build_ext】 cmd: {cmd}")
        os.system(cmd)

    def bdist_wheel(self):
        cmd = "python setup.py bdist_wheel"
        print(f"【bdist_wheel】 cmd: {cmd}")
        os.system(cmd)


def main():
    make = MakeInstaller()
    make.remove_build_dir()
    make.build_ext()
    make.bdist_wheel()


if __name__ == '__main__':
    main()
