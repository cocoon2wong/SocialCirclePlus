"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2024-06-05 16:10:57
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import qpid
import socialCircle
from qpid.mods import vis
from scripts import socialcircle_toy_example

TARGET_FILE = './README.md'


if __name__ == '__main__':
    qpid.help.update_readme(qpid.print_help_info(), TARGET_FILE)
