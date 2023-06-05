#!/usr/bin/env python
# coding=utf-8
# @Time    : 2021/12/20 14:32
# @Author  : 江斌
# @Software: PyCharm


from pathlib import Path


def get_all_whl_files(root: Path):
    whl_files = list(root.glob("*.whl"))
    whl_dict = {str(each).split('-')[1]: each.name for each in whl_files}
    return whl_dict


PROJECT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_DIR / "bulid"
DIST_DIR = PROJECT_DIR / "dist"

DIST_FILE_DICT = get_all_whl_files(DIST_DIR)

