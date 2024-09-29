---
layout: page
title: SocialCircle+
subtitle: "SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction"
cover-img: /subassets/img/head_pic.jpg
---
<!--
 * @Author: Ziqian Zou
 * @Date: 2024-05-31 15:53:21
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2024-09-29 10:02:52
 * @Description: file content
 * @Github: https://github.com/LivepoolQ
 * Copyright 2024 Ziqian Zou, All Rights Reserved.
-->

## Information

This is the homepage of our paper "SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction".

SocialCircle+ is an extended journal-version of our previous work [SocialCircle](https://github.com/cocoon2wong/SocialCircle). 
The paper is available on arXiv.
Click the buttons below for more information.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://arxiv.org/abs/2409.14984">📖 Paper</a>
    <!-- <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/SocialCirclePlus">📖 Supplemental Materials (TBA)</a>
    <br><br> -->
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/SocialCirclePlus">🛠️ Codes (PyTorch)</a>
    <a class="btn btn-colorful btn-lg" href="./guidelines">💡 Codes Guidelines</a>
    <br><br>
</div>

## Abstract

Trajectory prediction is a crucial aspect of understanding human behaviors.
Researchers have made efforts to represent socially interactive behaviors among pedestrians and utilize various networks to enhance prediction capability.
Unfortunately, they still face challenges not only in fully explaining and measuring how these interactive behaviors work to modify trajectories but also in modeling pedestrians' preferences to plan or participate in social interactions in response to the changeable physical environments as extra conditions.
This manuscript mainly focuses on the above explainability and conditionality requirements for trajectory prediction networks.
Inspired by marine animals perceiving other companions and the environment underwater by echolocation, this work constructs an angle-based conditioned social interaction representation SocialCircle+ to represent the socially interactive context and its corresponding conditions.
It employs a social branch and a conditional branch to describe how pedestrians are positioned in prediction scenes socially and physically in angle-based-cyclic-sequence forms.
Then, adaptive fusion is applied to fuse the above conditional clues onto the social ones to learn the final interaction representation.
Experiments demonstrate the superiority of SocialCircle+ with different trajectory prediction backbones.
Moreover, counterfactual interventions have been made to simultaneously verify the modeling capacity of causalities among interactive variables and the conditioning capability.

## Highlights

![SocialCirclePlus](./subassets/img/fig_method.png)

- The angle-based cyclic interaction modeling strategy and three SocialCircle meta components to represent the socially interactive context of each pedestrian;
- Three angle-based PhysicalCircle meta components to represent the physical environment around each prediction target as interaction conditions;
- The SocialCircle+ representation that is obtained by encoding and fusing the above physical components onto the social components in a partition-wise adaptive way, thus prompting trajectory prediction networks to learn to represent social interactions among pedestrians by taking into account physical environments as additional conditions.

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{wong2024socialcircle+,
  title={SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction},
  author={Wong, Conghao and Xia, Beihao and Zou, Ziqian and You, Xinge},
  journal={arXiv preprint arXiv:2409.14984},
  year={2024}
}
```

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn  
Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com
