# Unverified draft ideas from ChatGPT

import jax
import jax.numpy as jnp
from flax import nn
from typing import Tuple

class DownsampleBlock(nn.Module):
    """ Downsample block in U-Net model """

    channels: int

    def apply(self, x):
        x = nn.Conv(x, features=self.channels, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        x = nn.BatchNorm(x)
        x = nn.relu(x)
        x = nn.Conv(x, features=self.channels, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        x = nn.BatchNorm(x)
        x = nn.relu(x)
        x_pool = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
        return x, x_pool

class UpsampleBlock(nn.Module):
    """ Upsample block in U-Net model """

    channels: int

    def apply(self, x, x_skip):
        x = nn.ConvTranspose(x, features=self.channels, kernel_size=(3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
        x = nn.concatenate([x, x_skip], axis=-1)
        x = nn.Conv(x, features=self.channels, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        x = nn.BatchNorm(x)
        x = nn.relu(x)
        x = nn.Conv(x, features=self.channels, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        x = nn.BatchNorm(x)
        x = nn.relu(x)
        return x

class UNet(nn.Module):
    """ U-Net model for image segmentation """

    # # sample_size: Optional[Union[int, Tuple[int, int]]] = None
    # in_channels: int = 3
    # out_channels: int = 3
    # # center_input_sample: bool = False
    # # time_embedding_type: str = "positional"
    # # freq_shift: int = 0
    # # flip_sin_to_cos: bool = True
    # # down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    # # up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    # block_out_channels: Tuple[int] = (224, 448, 672, 896)
    # # layers_per_block: int = 2
    # # mid_block_scale_factor: float = 1
    # # downsample_padding: int = 1
    # # act_fn: str = "silu"
    # # attention_head_dim: Optional[int] = 8
    # norm_num_groups: int = 32
    # # norm_eps: float = 1e-5
    # # resnet_time_scale_shift: str = "default"
    # # add_attention: bool = True
    # # class_embed_type: Optional[str] = None
    # # num_class_embeds: Optional[int] = None

    channels: Tuple[int] = (64, 128, 256, 512, 1024)

    # in_channels=1, out_channels=1, block_out_channels=(16, 32, 64, 64), norm_num_groups=8

    def apply(self, x):
        x_downsamples = []
        for c in self.channels:
            x, x_pool = DownsampleBlock(c)(x)
            x_downsamples.append(x)
            x = x_pool
        x = nn.Conv(x, features=self.channels[-1]*2, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        x = nn.BatchNorm(x)
        x = nn.relu(x)
        x = nn.Conv(x, features=self.channels[-1], kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
        for c, x_skip in zip(reversed(self.channels[:-1]), reversed(x_downsamples)):
            x = UpsampleBlock(c)(x, x_skip)
        x = nn.Conv(x, features=1, kernel_size=(1, 1), strides=(1, 1), padding=[(0, 0), (0, 0)])
        x = nn.sigmoid(x)
        return x

#     def old(): PYTORCH CODE

#         time_embed_dim = block_out_channels[0] * 4

#         # Check inputs
#         if len(down_block_types) != len(up_block_types):
#             raise ValueError(
#                 f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
#             )

#         if len(block_out_channels) != len(down_block_types):
#             raise ValueError(
#                 f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
#             )

#         # input
#         self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))


#         self.down_blocks = nn.ModuleList([])
#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])

#         # down
#         output_channel = block_out_channels[0]
#         for i, down_block_type in enumerate(down_block_types):
#             input_channel = output_channel
#             output_channel = block_out_channels[i]
#             is_final_block = i == len(block_out_channels) - 1

#             down_block = get_down_block(
#                 down_block_type,
#                 num_layers=layers_per_block,
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 temb_channels=time_embed_dim,
#                 add_downsample=not is_final_block,
#                 resnet_eps=norm_eps,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attn_num_head_channels=attention_head_dim,
#                 downsample_padding=downsample_padding,
#                 resnet_time_scale_shift=resnet_time_scale_shift,
#             )
#             self.down_blocks.append(down_block)

#         # mid
#         self.mid_block = UNetMidBlock2D(
#             in_channels=block_out_channels[-1],
#             temb_channels=time_embed_dim,
#             resnet_eps=norm_eps,
#             resnet_act_fn=act_fn,
#             output_scale_factor=mid_block_scale_factor,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#             attn_num_head_channels=attention_head_dim,
#             resnet_groups=norm_num_groups,
#             add_attention=add_attention,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]
#             input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = get_up_block(
#                 up_block_type,
#                 num_layers=layers_per_block + 1,
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=prev_output_channel,
#                 temb_channels=time_embed_dim,
#                 add_upsample=not is_final_block,
#                 resnet_eps=norm_eps,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 attn_num_head_channels=attention_head_dim,
#                 resnet_time_scale_shift=resnet_time_scale_shift,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         # out
#         num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

# # Instantiate the model and compile it using JAX's jit function
# model = UNet()
# jit_model = jax.jit(model.apply)