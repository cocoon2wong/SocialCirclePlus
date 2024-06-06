"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-28 10:19:04
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.base import BaseManager


class BaseSocialCircleModel(BaseManager):

    def print_info(self, **kwargs):
        info: dict = {'SocialCircle+ Settings': None}

        if 'sc_args' in self.__dict__:
            factors = [item for item in ['velocity',
                                         'distance',
                                         'direction',
                                         'move_direction']
                       if getattr(self.sc_args, f'use_{item}')]

            info.update({
                # 'Transform type (SocialCircle)': self.sc_args.Ts,
                '- The number of circle partitions': self.sc_args.partitions,
                '- Maximum circle partitions': self.args.obs_frames,
                '- Factors in SocialCircle': factors,
            })

        if 'pc_args' in self.__dict__:
            info.update({
                '- Vision radiuses in PhysicalCircle': self.pc_args.vision_radius,
                '- Adaptive Fusion': 'Activated' if self.pc_args.adaptive_fusion
                else 'Disabled',
            })

        return super().print_info(**kwargs, **info)
