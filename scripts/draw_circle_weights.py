"""
@Author: Conghao Wong
@Date: 2024-05-16 15:08:43
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-28 10:32:34
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    FILE_NAME = sys.argv[1]
except:
    FILE_NAME = 'testsc'

# Put weights here
ATT_WEIGHTS = """
9.598438
24.592285
18.288754
15.323627
7.048789
""".split('\n')
ATT_WEIGHTS = [float(i) for i in ATT_WEIGHTS if len(i)]

COLOR_HIGH = (250, 100, 100)
COLOR_LOW = (60, 100, 220)

MAX_WIDTH = 0.3
MIN_WIDTH = 0.1


def cal_color(weights: list[float]):
    w: np.ndarray = np.array(weights)
    w = w - w.min()
    w /= (w.max() - w.min())
    color_high = np.array(COLOR_HIGH)
    color_low = np.array(COLOR_LOW)
    color = color_low + w[:, np.newaxis] * (color_high - color_low)
    return [(i.tolist()) for i in color/255]


def cal_radius(weights: list[float],
               max_value=MAX_WIDTH,
               min_value=MIN_WIDTH) -> np.ndarray:
    w: np.ndarray = np.array(weights)
    return (max_value - min_value) * w/w.max() + min_value


if __name__ == '__main__':

    fig = plt.figure()

    # Draw the big circle
    plt.pie(x=[1 for _ in ATT_WEIGHTS],
            radius=1,
            colors=cal_color(ATT_WEIGHTS))

    # Draw the center circle
    for index, r in enumerate(cal_radius(ATT_WEIGHTS)):
        colors: list = [(0, 0, 0, 0) for _ in ATT_WEIGHTS]
        colors[index] = fig.get_facecolor()
        plt.pie(x=[1 for _ in ATT_WEIGHTS],
                radius=1.0-r,
                colors=colors)

    plt.savefig(f := f'./{FILE_NAME}.png')
    plt.close()

    # Save as a png image
    fig_saved: np.ndarray = cv2.imread(f)
    alpha_channel = 255 * (fig_saved[:, :, 1] < 150)
    fig_png = np.concatenate(
        [fig_saved, alpha_channel[..., np.newaxis]], axis=-1)

    # Cut the image
    areas = fig_png[..., -1] == 255
    x_value = np.sum(areas, axis=0)
    x_index_all = np.where(x_value)[0]
    y_value = np.sum(areas, axis=1)
    y_index_all = np.where(y_value)[0]

    cv2.imwrite(f'./{FILE_NAME}_png.png',
                fig_png[y_index_all[0]:y_index_all[-1],
                        x_index_all[0]:x_index_all[-1]])
