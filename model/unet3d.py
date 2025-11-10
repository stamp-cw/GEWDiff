# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
# Copyright 2024 The ModelScope Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, deprecate, logging
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
from diffusers.models.unets.unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class UNet3DOutput(BaseOutput):
    """
    The output of [`UNet3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor

from diffusers.models.embeddings import GaussianFourierProjection
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal


       
class UNet3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-6,  # Increase eps to prevent division by zero
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        cross_attention_dim: int = 512,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = 16,
        time_cond_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError("The number of down_block_types and up_block_types does not match")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("The number of block_out_channels and down_block_types does not match")

        # Change the input layer to Conv3d and add safe initialization
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        # Use Xavier/Glorot initialization instead of Kaiming
        nn.init.xavier_uniform_(self.conv_in.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.conv_in.bias)
        if self.conv_in.bias is not None:
            nn.init.constant_(self.conv_in.bias, 0)
        num_attention_heads = [int(num_attention_heads)] * len(down_block_types)
        # Temporal embedding enhances stability
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LayerNorm(time_embed_dim, eps=norm_eps) ) # Add LayerNorm stable value

        # Category embedding repair
        if class_embed_type == "identity":
            self.class_embedding = nn.Identity()
        elif class_embed_type is not None:
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # Downsampling block
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=i != len(block_out_channels) - 1,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i] if num_attention_heads else None,
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1] if num_attention_heads else None,
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )

        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=i != len(block_out_channels) - 1,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i] if num_attention_heads else None,
                dual_cross_attention=False,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)

        # Output layer enhances stability
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=num_groups_out,
            eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], 
            out_channels, 
            kernel_size=3, 
            padding=1
        )
        nn.init.zeros_(self.conv_out.weight)  # Initialize to zero output
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DOutput, Tuple]:
        # Input Checking
        if torch.isnan(sample).any():
            raise ValueError("The input sample contains NaN values")

        # Centralized Input
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # Time processing
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        
        #Time Embedding
        emb = self.time_embedding(t_emb)
        if torch.isnan(emb).any():
            raise ValueError("Time embedding contains NaN values")

        # Category Conditions
        if self.class_embedding is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb

        # Forward propagation
        skip_sample = sample
        sample = self.conv_in(sample)
        
        # downsample_block
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            
            if torch.isnan(sample).any():
                raise ValueError("NaN values appear during downsampling")
            down_block_res_samples += res_samples

        # mid_block
        sample = self.mid_block(sample, emb)
        if torch.isnan(sample).any():
            raise ValueError("Intermediate block output contains NaN values")

        # up_blocks
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)
            
            if torch.isnan(sample).any():
                raise ValueError("NaN values appear during upsampling")

        # Output Processing
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample = sample + skip_sample[:,:sample.shape[1]]

        if not return_dict:
            return (sample,)

        return UNet3DOutput(sample=sample)