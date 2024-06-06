"""
@Author: Beihao Xia
@Date: 2023-03-20 16:15:25
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-29 21:04:31
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Beihao Xia, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure

from .__args import VArgs


class MinimalVModel(Model):
    """
    The `minimal` vertical model.
    It only includes the Transformer backbone when `args.T` is set to `none`.

    - considers nothing about interactions;
    - no keypoints-interpolation two-stage subnetworks;
    - contains only the backbone;
    - considers nothing about agents' multimodality.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set('output_pred_steps', 'all')
        self.v_args = self.args.register_subargs(VArgs, 'v_args')
        self.v_args._set_default('T', 'none')

        # Preprocess
        self.set_preprocess(**{PROCESS_TYPES.MOVE: 0})

        # Assign input and label types
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

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

        # Steps and shapes after applying transforms
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

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

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # Feature embedding and encoding -> (batch, obs, d/2)
        f_traj = self.te(obs)

        # Sampling random noise vectors
        all_predictions = []
        repeats = 1

        traj_targets = self.t1(obs)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_traj.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            # Transformer inputs -> (batch, steps, d)
            f_final = torch.concat([f_traj, f_z], dim=-1)

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


class MinimalV(Structure):
    """
    Training structure for the `minimal` vertical model.
    """
    MODEL_TYPE = MinimalVModel
