"""
@Author: Conghao Wong
@Date: 2023-08-15 20:30:51
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 13:52:15
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure

from .__args import PhysicalCircleArgs, SocialCircleArgs
from .__base import BaseSocialCircleModel
from .__layers import CircleFusionLayer, PhysicalCircleLayer, SocialCircleLayer
from .original_models import VArgs


class TransformerSCPlusModel(Model, BaseSocialCircleModel):
    """
    Transformer-SC+
    ---
    A simple Transformer-based trajectory prediction model.
    It takes the SocialCircle to model social interactions, and the
    PhysicalCircle to model interactions among agents and scene objects
    (physical interactions).
    Set the arg `--adaptive_fusion` when training this model to activate
    the adaptive fusion strategy to fuse SocialCircle and PhysicalCircle.

    NOTE:
    - It has no keypoints-interpolation two-stage subnetworks;
    - It contains only the Transformer backbone;
    - It considers nothing about agents' multimodality.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        from qpid.mods import segMaps

        # Init args
        self.args._set('output_pred_steps', 'all')
        self.v_args = self.args.register_subargs(VArgs, 'v_args')
        self.v_args._set_default('T', 'none')
        self.sc_args = self.args.register_subargs(SocialCircleArgs, 'sc')
        self.pc_args = self.args.register_subargs(PhysicalCircleArgs, 'pc')

        # Preprocess
        self.set_preprocess(**{PROCESS_TYPES.MOVE: 0})

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ,
                        segMaps.INPUT_TYPES.SEG_MAP,
                        segMaps.INPUT_TYPES.SEG_MAP_PARAS)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.v_args.T)

        # Transform layers
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.args.pred_frames, self.dim))

        # Trajectory embedding
        if type(self.t1) == layers.transfroms.NoneTransformLayer:
            self.te = layers.TrajEncoding(self.dim, self.d//2,
                                          torch.nn.Tanh)
        else:
            self.te = layers.TrajEncoding(self.dim, self.d//2,
                                          torch.nn.Tanh, self.t1)

        # SocialCircle (meta components) layer
        tslayer, _ = layers.get_transform_layers(self.sc_args.Ts)
        self.sc = SocialCircleLayer(partitions=self.sc_args.partitions,
                                    max_partitions=self.args.obs_frames,
                                    use_velocity=self.sc_args.use_velocity,
                                    use_distance=self.sc_args.use_distance,
                                    use_direction=self.sc_args.use_direction,
                                    relative_velocity=self.sc_args.rel_speed,
                                    use_move_direction=self.sc_args.use_move_direction)

        # PhysicalCircle (meta-compinents) layer
        self.pc = PhysicalCircleLayer(partitions=self.sc_args.partitions,
                                      max_partitions=self.args.obs_frames,
                                      use_velocity=self.sc_args.use_velocity,
                                      use_distance=self.sc_args.use_distance,
                                      use_direction=self.sc_args.use_direction,
                                      vision_radius=self.pc_args.vision_radius,
                                      pool_size=self.pc_args.seg_map_pool_size)

        # Fusion layer for SocialCircle and PhysicalCircle meta components
        self.spc = CircleFusionLayer(sclayer=self.sc,
                                     adaptive_fusion=self.pc_args.adaptive_fusion)

        # Encoding (embedding) layer for the fused circle
        self.ts = tslayer((self.args.obs_frames, self.sc.dim))
        self.tse = layers.TrajEncoding(self.sc.dim,
                                       self.d//2, torch.nn.ReLU,
                                       transform_layer=self.ts)

        # Concat and fuse SC
        self.concat_fc = layers.Dense(self.d, self.d//2, torch.nn.Tanh)

        # Steps and channels after applying transforms
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape
        self.Tsteps_en = max(self.Tsteps_en, self.sc_args.partitions)

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        self.ms_fc = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d, 2*self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(2*self.d, self.Tchannels_de)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # Compute SocialCircle and PhysicalCircle
        social_circle = self.sc.implement(self, inputs)
        physical_circle = self.pc.implement(self, inputs)

        # Fuse SocialCircles and PhysicalCircles
        sp_circle = self.spc(social_circle, physical_circle)

        # Encode the final SocialCircle+
        f_social = self.tse(sp_circle)    # (batch, steps, d/2)

        # feature embedding and encoding -> (batch, obs, d)
        f_traj = self.te(obs)

        # Feature fusion
        f_traj = self.sc.pad(f_traj)
        f_behavior = torch.concat([f_traj, f_social], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Sampling random noise vectors
        all_predictions = []
        repeats = 1

        traj_targets = self.t1(obs)
        traj_targets = self.sc.pad(traj_targets)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            # Transformer inputs -> (batch, steps, d)
            f_final = torch.concat([f_behavior, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Generations -> (batch, pred_steps, d)
            adj = self.ms_fc(f_final)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, pred_steps, d)

            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)

            y = self.it1(y)
            all_predictions.append(y)

        return torch.concat(all_predictions, dim=-3)   # (batch, 1, pred, dim)


class TransformerSCPlusStructure(Structure):
    MODEL_TYPE = TransformerSCPlusModel
