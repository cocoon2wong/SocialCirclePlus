"""
@Author: Conghao Wong
@Date: 2024-05-27 20:04:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 13:01:08
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure, loss

from .__args import PhysicalCircleArgs, SocialCircleArgs
from .__base import BaseSocialCircleModel
from .__layers import CircleFusionLayer, PhysicalCircleLayer, SocialCircleLayer
from .original_models import VArgs


class MSNSCPlusModel(Model, BaseSocialCircleModel):
    """
    MSN-SC+
    ---
    `MSN` Model with SocialCircle+.

    This model comes from "Msn: multi-style network for trajectory prediction".
    Its original interaction-modeling part has been removed, and layers
    related to SocialCircle and PhysicalCircle are plugged in.
    Set the arg `--adaptive_fusion` when training this model to activate
    the adaptive fusion strategy to fuse SocialCircle and PhysicalCircle.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        from qpid.mods import contextMaps, segMaps
        from qpid.mods.contextMaps.settings import (MAP_HALF_SIZE,
                                                    POOLING_BEFORE_SAVING)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', str(self.args.pred_frames - 1))

        self.v_args = self.args.register_subargs(VArgs, 'v_args')
        self.v_args._set('T', 'none')

        self.sc_args = self.args.register_subargs(SocialCircleArgs, 'sc')
        self.pc_args = self.args.register_subargs(PhysicalCircleArgs, 'pc')

        self.structure: Structure
        self.structure.metrics.set({loss.FDE: 1.0})

        # Set preprocesser
        preprocess = []
        for index, operation in enumerate(["NONE",
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess, **{PROCESS_TYPES.MOVE: 0})

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        contextMaps.INPUT_TYPES.MAP,
                        INPUT_TYPES.NEIGHBOR_TRAJ,
                        segMaps.INPUT_TYPES.SEG_MAP,
                        segMaps.INPUT_TYPES.SEG_MAP_PARAS)

        # Map parameters
        self.MAP_HALF_SIZE = MAP_HALF_SIZE
        self.POOLING_BEFORE_SAVING = POOLING_BEFORE_SAVING

        # Layers
        # context feature
        if not POOLING_BEFORE_SAVING:
            self.average_pooling = layers.MaxPooling2D((5, 5))

        self.flatten = layers.Flatten(2)
        self.context_dense1 = layers.Dense(
            ((MAP_HALF_SIZE*2)//5)**2,
            self.args.obs_frames * 64,
            activation=torch.nn.Tanh)

        # traj embedding
        self.pos_embedding = layers.Dense(2, 64, torch.nn.Tanh)

        # SocialCircle (meta components) layer
        tslayer, _ = layers.get_transform_layers(self.sc_args.Ts)
        self.sc = SocialCircleLayer(partitions=self.sc_args.partitions,
                                    max_partitions=self.args.obs_frames,
                                    use_velocity=self.sc_args.use_velocity,
                                    use_distance=self.sc_args.use_distance,
                                    use_direction=self.sc_args.use_direction,
                                    relative_velocity=self.sc_args.rel_speed,
                                    use_move_direction=self.sc_args.use_move_direction)

        # PhysicalCircle (meta components) layer
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
        self.tse = layers.TrajEncoding(self.sc.dim, 64, torch.nn.ReLU,
                                       transform_layer=self.ts)

        # Concat and fuse SC
        self.concat_fc = layers.Dense(128, 64, torch.nn.Tanh)

        # Transformer is used as a feature extractor
        self.T1 = transformer.Transformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            dff=512,
            input_vocab_size=2,
            target_vocab_size=2,
            pe_input=self.args.obs_frames,
            pe_target=self.args.obs_frames,
            include_top=False
        )

        # Trainable adj matrix and gcn layer
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(128, self.v_args.Kc, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(128, 128)

        # decoder
        self.decoder = layers.Dense(128, 2)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        from qpid.mods import contextMaps

        # Unpack inputs
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        maps = self.get_input(inputs, contextMaps.INPUT_TYPES.MAP)

        # Compute SocialCircle and PhysicalCircle
        social_circle = self.sc.implement(self, inputs)
        physical_circle = self.pc.implement(self, inputs)

        # Fuse SocialCircles and PhysicalCircles
        sp_circle = self.spc(social_circle, physical_circle)

        # Encode the final SocialCircle+
        f_social = self.tse(sp_circle)      # (batch, steps, d/2)

        # Traj embedding, out shape == (batch, obs, 64)
        f_traj = self.pos_embedding(obs)

        # Feature fusion
        f_traj = self.sc.pad(f_traj)
        f_behavior = torch.concat([f_traj, f_social], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Encoding context maps into context features
        # Shape of maps is (batch, 100, 100)
        # context feature, shape == (batch, obs, 64)
        if not self.POOLING_BEFORE_SAVING:
            average_pooling = self.average_pooling(maps[:, None])
        else:
            average_pooling = maps

        f_flatten = self.flatten(average_pooling)
        f_context = self.context_dense1(f_flatten)
        f_context = torch.reshape(f_context, [-1, self.args.obs_frames, 64])

        # Concat all features, shape == (batch, obs, 128)
        concat_feature = torch.concat([f_behavior, f_context], dim=-1)

        # Transformer output shape is (batch, obs, 128)
        f_tran, _ = self.T1(inputs=concat_feature,
                            targets=obs,
                            training=training)

        # Multiple generations
        adj = self.ms_fc(concat_feature)        # (batch, obs, pred)
        adj = torch.transpose(adj, -1, -2)
        f_multi = self.ms_conv(f_tran, adj)     # (batch, Kc, 128)

        # Forecast destinations
        predictions = self.decoder(f_multi)         # (batch, Kc, 2)
        return predictions[..., None, :]


class MSNSCPlusStructure(Structure):
    MODEL_TYPE = MSNSCPlusModel
