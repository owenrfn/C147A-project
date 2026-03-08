from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import SpectrogramNorm
from emg2qwerty.transforms import Transform

class SpectrogramToImage(nn.Module):
    """
    Converts spectrogram tensor from (T, N, B, C, F) to (N, B*C, F, T).
    """

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.permute(1, 2, 3, 4, 0).contiguous()
        sample = sample.flatten(start_dim=1, end_dim=2)
        return sample


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.relu(sample + self.block(sample))
    

class CNNRNNEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        proj_dim: int,
        cnn_output_size: int,
        rnn_hidden_size: int,
        rnn_layers: int,
        rnn_dropout: float,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1)),

            ResidualBlock(128)
        )

        self.pre_rnn = nn.Linear(cnn_output_size, proj_dim)

        self.rnn = nn.GRU(
            input_size = proj_dim,
            hidden_size = rnn_hidden_size,
            num_layers = rnn_layers,
            dropout = rnn_dropout,
            bidirectional = True,
            batch_first = False,
        )

        self.out_features = 2 * rnn_hidden_size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.cnn(tensor)
        out = out.permute(3, 0, 1, 2).contiguous()

        t, n, c, f = out.shape
        out = out.view(t, n, c * f)

        out = self.pre_rnn(out)

        out, _ = self.rnn(out)
        return out
    

class CNNRNN(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        proj_dim: int = 512,
        cnn_output_size = 2048,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        self.spec_norm = SpectrogramNorm(channels=num_channels)
        self.to_image = SpectrogramToImage()

        self.encoder = CNNRNNEncoder(
            channels = num_channels,
            proj_dim = proj_dim,
            cnn_output_size = cnn_output_size,
            rnn_hidden_size = rnn_hidden_size,
            rnn_layers = rnn_layers,
            rnn_dropout = rnn_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.spec_norm(inputs)
        # (T, N, B, C, F)

        out = self.to_image(out)
        # (N, 32, 33, T)

        out = self.encoder(out)
        # (T, N, 2 * hidden_size)

        out = self.classifier(out)
        # (T, N, num_classes)

        return out
    

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )