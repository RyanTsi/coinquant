"""Feature extractors for high-dimensional rolling trading observations.

The default SB3 ``MlpPolicy`` treats a flattened market window as an unordered
vector.  ``TemporalFeatureExtractor`` keeps the temporal structure intact: a
small causal convolution and GRU encode the market window, while a separate
GRU encodes the account-state history.  The two representations are fused by a
compact projection before the actor/critic heads see them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalFeatureExtractor(BaseFeaturesExtractor):
    """Encode market and account histories separately before fusion.

    The observation contract is the one emitted by :class:`ObservationBuilder`:
    flattened market rows followed by flattened account rows.  Dimensions are
    passed explicitly from the training frame so this extractor also works with
    custom feature sets and account history lengths.
    """

    def __init__(
        self,
        observation_space: Any,
        *,
        market_feature_count: int,
        window_size: int,
        account_feature_count: int,
        account_history_length: int,
        conv_channels: int = 64,
        recurrent_hidden_size: int = 128,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)
        dimensions = (
            market_feature_count,
            window_size,
            account_feature_count,
            account_history_length,
            conv_channels,
            recurrent_hidden_size,
            features_dim,
        )
        if any(int(value) != value or int(value) <= 0 for value in dimensions):
            raise ValueError("temporal extractor dimensions must be positive integers")
        self.market_feature_count = int(market_feature_count)
        self.window_size = int(window_size)
        self.account_feature_count = int(account_feature_count)
        self.account_history_length = int(account_history_length)
        market_size = self.market_feature_count * self.window_size
        account_size = self.account_feature_count * self.account_history_length
        expected_size = market_size + account_size
        if int(observation_space.shape[0]) != expected_size:
            raise ValueError(
                f"observation size {observation_space.shape[0]} does not match "
                f"market/account dimensions ({expected_size})"
            )

        # Conv1d sees feature channels and slides over bars.  LayerNorm after
        # the recurrent states keeps the fused representation well-conditioned
        # for both PPO and SAC.
        self.market_encoder = nn.Sequential(
            nn.Conv1d(self.market_feature_count, int(conv_channels), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(int(conv_channels), int(conv_channels), kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.market_gru = nn.GRU(
            input_size=int(conv_channels),
            hidden_size=int(recurrent_hidden_size),
            batch_first=True,
        )
        self.account_gru = nn.GRU(
            input_size=self.account_feature_count,
            hidden_size=max(32, int(recurrent_hidden_size) // 2),
            batch_first=True,
        )
        account_hidden = max(32, int(recurrent_hidden_size) // 2)
        self.fusion = nn.Sequential(
            nn.Linear(int(recurrent_hidden_size) + account_hidden, int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.SiLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = observations.float()
        batch_size = values.shape[0]
        market_size = self.market_feature_count * self.window_size
        market = values[:, :market_size].reshape(
            batch_size, self.window_size, self.market_feature_count
        )
        # [batch, bars, channels] -> [batch, channels, bars]
        market = self.market_encoder(market.transpose(1, 2)).transpose(1, 2)
        _, market_state = self.market_gru(market)
        market_state = market_state[-1]

        account = values[:, market_size:].reshape(
            batch_size, self.account_history_length, self.account_feature_count
        )
        _, account_state = self.account_gru(account)
        account_state = account_state[-1]
        return self.fusion(torch.cat((market_state, account_state), dim=1))
