# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from emg2qwerty.transforms import ChannelDropout


def test_channel_dropout_noop_with_zero_probability():
    x = torch.randn(100, 2, 16)
    y = ChannelDropout(dropout_prob=0.0)(x)
    assert torch.equal(x, y)


def test_channel_dropout_keeps_at_least_one_channel():
    torch.manual_seed(0)
    x = torch.ones(20, 16)
    y = ChannelDropout(dropout_prob=1.0)(x)

    # Exactly one channel should survive when p=1.0
    active_channels = (y.abs().sum(dim=0) > 0).sum().item()
    assert active_channels == 1
