# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.modules import CNNEncoder


def test_cnn_encoder_shape():
    x = torch.randn(200, 4, 528)
    encoder = CNNEncoder(num_features=528, num_blocks=3, kernel_width=7)
    y = encoder(x)
    assert y.shape == x.shape
