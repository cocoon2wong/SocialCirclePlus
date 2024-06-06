"""
@Author: Conghao Wong
@Date: 2021-04-16 16:02:45
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-25 16:41:52
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import qpid
import socialCircle


def get_menmber(package):
    return [item for _, item in inspect.getmembers(package) if (inspect.isclass(item) or inspect.ismodule(item)) and (not item.__name__.startswith('_'))]


def print_all_members(package):
    waiting = get_menmber(package)
    name_list = []
    lens = ['```mermaid\n',
            '    graph LR\n']

    while len(waiting):
        item = waiting.pop(-1)
        if inspect.ismodule(item):
            if item.__name__.split('.')[0] in [package.__name__]:
                waiting += get_menmber(item)
        else:
            name = item.__name__
            module = item.__module__
            class_name = f'{module}/{name}'
            info = f'{module}_{name}'

            if name == 'builtins':
                continue

            if class_name in name_list:
                continue

            for father in item.__bases__:
                father_name = father.__name__
                father_module = father.__module__
                father_info = f'{father_module}_{father_name}'

                print(f'{father_info} -> {class_name}')
                lens.append(f'        {father_info}("{father_name}({father_module})") ' +
                            f'--> {info}("{name}({module})")\n')

                if father_name != 'object':
                    waiting += get_menmber(item)

            name_list.append(class_name)

    lens.append('```\n')
    return lens


def update_readme_file(file_path, new_lines, start: str, end: str):
    with open(file_path, 'r') as f:
        all_lines = f.readlines()

    start_line = -1
    end_line = -1
    for index, line in enumerate(all_lines):
        if line.startswith(start):
            start_line = index

        if line.startswith(end) and start_line > 0:
            end_line = index
            break

    if start_line > 0 and end_line > 0:
        write_lines = all_lines[:start_line+1] + \
            new_lines + all_lines[end_line:]
        with open(file_path, 'w+') as f:
            f.writelines(write_lines)

        print('File update success.')


lines_qpid = print_all_members(qpid)
lines_sc = print_all_members(socialCircle)


update_readme_file('./classRef.md', lines_qpid + lines_sc,
                   start='<!-- GRAPH BEGINS HERE -->',
                   end='<!-- GRAPH ENDS HERE -->')
