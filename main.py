"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 11:14:08
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import torch

import qpid


def main(args: list[str], run_train_or_test=True):
    min_args = qpid.args.Args(args, is_temporary=True)

    if (h := min_args.help) != 'null':
        qpid.print_help_info('all_args' if h == 'True' else h)
        exit()

    t = qpid.get_structure(min_args.model)(args)

    if run_train_or_test:
        t.train_or_test()

    # It is used to debug
    if t.args.verbose:
        t.print_info_all()

    return t


if __name__ == '__main__':
    import socialCircle
    torch.autograd.set_detect_anomaly(True)
    main(sys.argv)
