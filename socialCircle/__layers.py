"""
@Author: Conghao Wong
@Date: 2023-08-08 14:55:56
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 13:49:26
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, process
from qpid.utils import get_mask

from .__args import PhysicalCircleArgs

NORMALIZED_SIZE = None

INF = 1000000000
SAFE_THRESHOLDS = 0.05
MU = 0.00000001


class SocialCircleLayer(torch.nn.Module):
    """
    A layer to compute SocialCircle meta components.

    Supported factors:
    - Velocity;
    - Distance;
    - Direction;
    - Movement Direction (Optional).
    """

    def __init__(self, partitions: int,
                 max_partitions: int,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_direction: bool | int = True,
                 use_move_direction: bool | int = False,
                 mu=0.0001,
                 relative_velocity: bool | int = False,
                 *args, **kwargs):
        """
        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param max_partitions: The number of partitions (after zero padding).

        ## SocialCircle Meta Components
        :param use_velocity: Choose whether to use the `velocity` factor.
        :param use_distance: Choose whether to use the `distance` factor.
        :param use_direction: Choose whether to use the `direction` factor.
        :param use_move_direction: Choose whether to use the `move direction` factor.

        ## SocialCircle Options
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing. \
            It only works when `relative_velocity` is set to `True`.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.rel_velocity = relative_velocity
        self.use_move_direction = use_move_direction
        self.mu = mu

    @property
    def dim(self) -> int:
        """
        The number of SocialCircle factors.
        """
        return int(self.use_velocity) + int(self.use_distance) + \
            int(self.use_direction) + int(self.use_move_direction)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        # Velocity factor
        if self.use_velocity:
            # Calculate velocities
            nei_velocity = torch.norm(nei_vector, dim=-1)    # (batch, n)
            obs_velocity = torch.norm(obs_vector, dim=-1)    # (batch, 1)

            # Speed factor in the SocialCircle
            if self.rel_velocity:
                f_velocity = (nei_velocity + self.mu)/(obs_velocity + self.mu)
            else:
                f_velocity = nei_velocity

        # Distance factor
        if self.use_distance:
            f_distance = torch.norm(nei_posion_vector, dim=-1)

        # Move direction factor
        if self.use_move_direction:
            obs_move_direction = torch.atan2(obs_vector[..., 0],
                                             obs_vector[..., 1])
            nei_move_direction = torch.atan2(nei_vector[..., 0],
                                             nei_vector[..., 1])
            delta_move_direction = nei_move_direction - obs_move_direction
            f_move_direction = delta_move_direction % (2*np.pi)

        # Direction factor
        f_direction = torch.atan2(nei_posion_vector[..., 0],
                                  nei_posion_vector[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_trajs, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        social_circle = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001
            social_circle.append([])

            if self.use_velocity:
                _velocity = torch.sum(f_velocity * _mask, dim=-1) / n
                social_circle[-1].append(_velocity)

            if self.use_distance:
                _distance = torch.sum(f_distance * _mask, dim=-1) / n
                social_circle[-1].append(_distance)

            if self.use_direction:
                _direction = torch.sum(f_direction * _mask, dim=-1) / n
                social_circle[-1].append(_direction)

            if self.use_move_direction:
                _move_d = torch.sum(f_move_direction * _mask, dim=-1) / n
                social_circle[-1].append(_move_d)

        # Shape of the final SocialCircle: (batch, p, 3)
        social_circle = [torch.stack(i) for i in social_circle]
        social_circle = torch.stack(social_circle)
        social_circle = torch.permute(social_circle, [2, 0, 1])
        social_circle = self.pad(social_circle)
        return social_circle

    def implement(self, model: Model, inputs: list[torch.Tensor]):
        """
        Compute the SocialCircle from original model inputs.
        """
        # Unpack inputs
        # (batch, obs, dim)
        obs = model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, a:=max_agents, obs, dim)
        nei = model.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Start computing the SocialCircle
        # SocialCircle will be computed on each agent's center point
        c_obs = model.picker.get_center(obs)[..., :2]
        c_nei = model.picker.get_center(nei)[..., :2]

        # Compute and encode the SocialCircle
        social_circle = self(c_obs, c_nei)
        return social_circle

    def pad(self, input: torch.Tensor):
        """
        Zero-padding the input tensor (whose shape must be `(batch, steps, dim)`).
        It will pad the input tensor on the `steps` axis if `steps < max_partitions`,
        where the `max_partitions` is usually the maximum one of either the number of
        observation steps or the number of SocialCircle partitions.
        """
        current_steps = input.shape[-2]
        target_steps = max(self.max_partitions, self.partitions)
        if ((p := target_steps - current_steps) > 0):
            paddings = [0, 0, 0, p, 0, 0]
            return torch.nn.functional.pad(input, paddings)
        else:
            return input


class PhysicalCircleLayer(torch.nn.Module):
    """
    A layer to compute PhysicalCircle Meta components.

    Supported factors:
    - Relative Velocity;
    - Minimum Distance;
    - Direction.
    """

    def __init__(self, partitions: int,
                 max_partitions: int,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_direction: bool | int = True,
                 vision_radius: float = 2.0,
                 pool_size: int = -1,
                 *args, **kwargs):
        """
        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param max_partitions: The number of partitions (after zero padding).

        ## PhysicalCircle Meta Components
        :param use_velocity: Choose whether to use the `relative velocity` factor.
        :param use_distance: Choose whether to use the `minimum distance` factor.
        :param use_direction: Choose whether to use the `direction` factor.

        ## PhysicalCircle Options
        :param vision_radius: The raduis to compute PhysicalCircle meta components \
            on the scene segmentation map. It should be a float number that used to \
            represent the scaling relationship between the field of vision and the \
            length of the target agent's movement (during the observation period). \
            For example, suppose that someone has moved 10 meters during the observation \
            period, and given the `vision_radius = 2.0`, its radius to compute on the \
            segmentation map will be `2.0 * 10 = 20` meters.
        :param pool_size: The kerne size of maxpooling operation on the original \
            segmentation maps. It is used to speed up the model inference. Set \
            it to `-1` do disable the pooling operation.
        """

        super().__init__(*args, **kwargs)

        from qpid.mods.segMaps.settings import NORMALIZED_SIZE

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.radius = vision_radius

        # Compute all pixels' indices
        pool_size = 1 if pool_size == -1 else pool_size
        xs, ys = torch.meshgrid(torch.arange(NORMALIZED_SIZE//pool_size),
                                torch.arange(NORMALIZED_SIZE//pool_size),
                                indexing='ij')
        self.map_pos_pixel = torch.stack(
            [xs.reshape([-1]), ys.reshape([-1])], dim=-1).to(torch.float32)
        self.map_pos_pixel = self.map_pos_pixel * pool_size + pool_size // 2

        if pool_size > 1:
            self.pool = torch.nn.MaxPool2d((pool_size, pool_size))
        else:
            self.pool = None

    @property
    def dim(self) -> int:
        """
        The number of PhysicalCircle factors.
        """
        return (int(self.use_velocity) +
                int(self.use_distance) +
                int(self.use_direction))

    def forward(self, seg_maps: torch.Tensor,
                seg_map_paras: torch.Tensor,
                trajectories: torch.Tensor,
                current_pos: torch.Tensor,
                *args, **kwargs):

        # Move back to original trajectories
        _obs = trajectories + current_pos

        # Treat seg maps as a long sequence
        if self.pool:
            _maps = self.pool(seg_maps[..., None, :, :])[..., 0, :, :]
        else:
            _maps = seg_maps

        _maps = torch.flatten(_maps, start_dim=1, end_dim=-1)
        map_safe_mask = (_maps <= SAFE_THRESHOLDS).to(torch.float32)

        # Compute velocity (moving length) during observation period
        moving_vector = _obs[..., -1, :] - _obs[..., 0, :]
        moving_length = torch.norm(moving_vector, dim=-1)   # (batch)

        # Compute pixel positions on seg maps
        W = seg_map_paras[..., :2][..., None, :]
        b = seg_map_paras[..., 2:4][..., None, :]

        # Compute angles and distances
        self.map_pos_pixel = self.map_pos_pixel.to(W.device)
        map_pos = (self.map_pos_pixel - b) / W  # (batch, a*a, 2)

        # Compute distances and angles of all pixels
        direction_vectors = map_pos - current_pos           # (batch, a*a, 2)
        distances = torch.norm(direction_vectors, dim=-1)   # (batch, a*a)

        angles = torch.atan2(direction_vectors[..., 0],
                             direction_vectors[..., 1])     # (batch, a*a)
        angle_indices = (angles % (2*np.pi)) / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Compute the `equivalent` distance
        equ_dis = (distances + MU) / (_maps + MU)

        # Compute the vision range
        r = (self.radius * moving_length)[..., None]
        radius_mask = (distances <= r).to(torch.float32)

        # Compute the PhysicalCircle
        pc = []
        for ang in range(self.partitions):
            # Compute the partition's mask
            angle_mask = (angle_indices == ang).to(torch.float32)
            final_mask = radius_mask * angle_mask

            # Compute the minimum distance factor
            d = (0 * map_safe_mask +
                 (1 - map_safe_mask) * final_mask * (equ_dis))

            # Find the non-zero minimum value
            zero_mask = (d == 0).to(torch.float32)
            d = (torch.ones_like(d) * zero_mask * INF +
                 d * (1 - zero_mask))
            min_d, _ = torch.min(d, dim=-1)

            # `d == INF` <=> there are no obstacles
            obstacle_mask = (min_d < INF).to(torch.float32)

            # The velocity factor
            if self.use_velocity:
                f_velocity = moving_length * obstacle_mask
                pc.append(f_velocity)

            # The distance factor
            if self.use_distance:
                f_min_distance = min_d * obstacle_mask
                pc.append(f_min_distance)

            # The direction factor
            if self.use_direction:
                _angle = 2 * np.pi * (ang + 0.5) / self.partitions
                f_direction = _angle * obstacle_mask
                pc.append(f_direction)

        # Final return shape: (batch, max_partitions, dim)
        pc = torch.stack(pc, dim=-1)
        pc = pc.reshape([-1, self.partitions, self.dim])

        if (((m := self.max_partitions) is not None) and
                (m > (n := self.partitions))):
            paddings = [0, 0, 0, m - n, 0, 0]
            pc = torch.nn.functional.pad(pc, paddings)

        return pc

    def implement(self, model: Model, inputs: list[torch.Tensor]):
        """
        Compute the Physical from original model inputs.
        """
        from qpid.mods import segMaps

        # Unpack inputs
        # (batch, obs, dim)
        obs = model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # Segmentaion-map-related inputs (to compute the PhysicalCircle)
        # (batch, h, w)
        seg_maps = model.get_input(inputs, segMaps.INPUT_TYPES.SEG_MAP)

        # (batch, 4)
        seg_map_paras = model.get_input(
            inputs, segMaps.INPUT_TYPES.SEG_MAP_PARAS)

        # Process model inputs
        pc_args = model.args.register_subargs(PhysicalCircleArgs, 'pc')
        if pc_args.use_empty_seg_maps:
            seg_maps = torch.zeros_like(seg_maps)

        # Get unprocessed positions from the `MOVE` layer
        if (m_layer := model.processor.get_layer_by_type(process.Move)):
            unprocessed_pos = m_layer.ref_points
        else:
            unprocessed_pos = torch.zeros_like(obs[..., -1:, :])

        # Start computing the PhysicalCircle
        # PhysicalCircle will be computed on each agent's 2D center point
        c_obs = model.picker.get_center(obs)[..., :2]
        c_unpro_pos = model.picker.get_center(unprocessed_pos)[..., :2]

        # Compute PhysicalCircle meta components
        physical_circle = self(seg_maps, seg_map_paras, c_obs, c_unpro_pos)

        # Rotate the PhysicalCircle (if needed)
        if (r_layer := model.processor.get_layer_by_type(process.Rotate)):
            physical_circle = self.rotate(physical_circle, r_layer.angles)

        return physical_circle

    def rotate(self, circle: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate the physicalCircle. (Usually used after preprocess operations.)
        """
        # Rotate the circle <=> left or right shift the circle
        # Compute shift length
        angles = angles % (2*np.pi)
        partition_angle = (2*np.pi) / (self.partitions)
        move_length = (angles // partition_angle).to(torch.int32)

        # Remove paddings
        valid_circle = circle[..., :self.partitions, :]
        valid_circle = torch.concat([valid_circle, valid_circle], dim=-2)
        paddings = circle[..., self.partitions:, :]

        # Shift each circle
        rotated_circles = []
        for _circle, _move in zip(valid_circle, move_length):
            rotated_circles.append(_circle[_move:self.partitions+_move])

        rotated_circles = torch.stack(rotated_circles, dim=0)
        return torch.concat([rotated_circles, paddings], dim=-2)


class CircleFusionLayer(torch.nn.Module):
    """
    A layer to fuse SocialCircle and PhysicalCircle meta components,
    i.e., to compute SocialCircle+ meta components.
    """

    def __init__(self, sclayer: SocialCircleLayer,
                 adaptive_fusion: bool | int = False,
                 feature_dimension=128,
                 *args, **kwargs):
        """
        :param sclayer: The `SocialCircleLayer` object. It is used as a \
            reference for SocialCircle and PhysicalCircle settings.
        :param adaptive_fusion: Choose whether to activate the adaptive \
            fusion strategy when fusing meta components from two circles.
        :param feature_dimension: Feature dimension of the encoded circle \
            meta components. It only works when `adaptive_fusion == True`.
        """

        super().__init__(*args, **kwargs)

        self.max_partitions = sclayer.max_partitions
        self.use_velocity = sclayer.use_velocity
        self.use_distance = sclayer.use_distance
        self.use_direction = sclayer.use_direction

        self.d = feature_dimension
        self.adaptive_fusion = adaptive_fusion

        if self.adaptive_fusion:
            self.fc1 = layers.Dense(sclayer.dim, self.d, torch.nn.Tanh)
            self.fc2 = layers.Dense(self.d, 1, torch.nn.Sigmoid)

    def forward(self, sc: torch.Tensor, pc: torch.Tensor, *args, **kwargs):

        index = -1
        spc = []

        if self.adaptive_fusion:
            w_partition_sc = self.fc2(self.fc1(sc))     # (batch, p, 1)
            w_partition_pc = self.fc2(self.fc1(pc))     # (batch, p, 1)

            return ((w_partition_sc * sc + w_partition_pc * pc) /
                    (w_partition_sc + w_partition_pc + MU))

        if self.use_velocity:
            index += 1
            f_v_sc = sc[..., index]
            f_v_pc = pc[..., index]
            f_v_spc = torch.max(f_v_sc, f_v_pc)
            spc.append(f_v_spc)

        if self.use_distance:
            index += 1
            f_dis_sc = sc[..., index]
            f_dis_pc = pc[..., index]
            _cross_mask = self.get_cross_mask(f_dis_sc, f_dis_pc)
            f_dis_spc = (_cross_mask * torch.min(f_dis_sc, f_dis_pc) +
                         (1 - _cross_mask) * torch.max(f_dis_sc, f_dis_pc))
            spc.append(f_dis_spc)

        if self.use_direction:
            index += 1
            f_dir_sc = sc[..., index]
            f_dir_pc = pc[..., index]
            _cross_mask = self.get_cross_mask(f_dir_sc, f_dir_pc)
            f_dir_spc = (_cross_mask * 0.5 * (f_dir_sc + f_dir_pc) +
                         (1 - _cross_mask) * torch.max(f_dir_sc, f_dir_pc))
            spc.append(f_dir_spc)

        spc = torch.stack(spc, dim=-1)
        return spc

    def get_cross_mask(self, x: torch.Tensor, y: torch.Tensor):
        x_mask = (x > MU).to(torch.float32)
        y_mask = (y > MU).to(torch.float32)
        return x_mask * y_mask
