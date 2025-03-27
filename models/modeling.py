# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torch.nn.init import trunc_normal_

import models.configs as configs
from .modeling_resnet import ResNetV2

logger = logging.getLogger(__name__)

# Constants for weight loading from pretrained models
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Convert numpy array weights to PyTorch tensor, optionally transposing for conv layers.

    Args:
        weights (np.ndarray): Numpy array containing weights
        conv (bool): If True, transpose weights from HWIO to OIHW format for conv layers

    Returns:
        torch.Tensor: Converted weights as PyTorch tensor
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    """Swish activation function: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


# Activation function mapping
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    """Multi-head attention module from Transformer architecture."""

    def __init__(self, config, vis):
        """Initialize attention module.

        Args:
            config: Model configuration object containing hyperparameters
            vis (bool): Whether to return attention weights for visualization
        """
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear transformations for query, key, value
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # Output projection and dropout
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """Reshape and transpose input for multi-head attention computation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """Forward pass for attention layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            tuple: (attention_output, attention_weights) where attention_weights is None if not self.vis
        """
        # Project inputs to query, key, value
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Project back to hidden size
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    """Feed-forward network (MLP) module for Transformer."""

    def __init__(self, config):
        """Initialize MLP module.

        Args:
            config: Model configuration object containing hyperparameters
        """
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]  # Using GELU activation
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        """Forward pass for MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after MLP transformation
        """
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch and position embeddings.

    Can use either pure transformer or hybrid (CNN + transformer) approach.
    """

    def __init__(self, config, img_size, in_channels=3):
        """Initialize embeddings.

        Args:
            config: Model configuration object
            img_size (int or tuple): Input image size
            in_channels (int): Number of input channels (default: 3 for RGB)
        """
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)  # Ensure img_size is a tuple

        # Handle grid-based patches (for hybrid models) or regular patches
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        # Hybrid model uses ResNet for initial feature extraction
        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor
            )
            in_channels = self.hybrid_model.width * 16

        # Patch embeddings projection
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Position embeddings and class token
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # Initialize weights
        trunc_normal_(self.position_embeddings, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        """Forward pass for embeddings.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch, channels, height, width]

        Returns:
            torch.Tensor: Combined embeddings of shape [batch, n_patches+1, hidden_size]
        """
        B = x.shape[0]  # Batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand class token for batch

        # Hybrid model first processes through ResNet
        if self.hybrid:
            x = self.hybrid_model(x)

        # Create patch embeddings
        x = self.patch_embeddings(x)  # [B, hidden_size, grid_h, grid_w]
        x = x.flatten(2)  # [B, hidden_size, n_patches]
        x = x.transpose(-1, -2)  # [B, n_patches, hidden_size]

        # Concatenate class token
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, hidden_size]

        # Add position embeddings
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    """Transformer block consisting of attention and MLP layers with residual connections."""

    def __init__(self, config, vis):
        """Initialize transformer block.

        Args:
            config: Model configuration object
            vis (bool): Whether to return attention weights for visualization
        """
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        """Forward pass for transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            tuple: (output_tensor, attention_weights)
        """
        # Attention sub-layer
        h = x  # Residual connection
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # Add residual

        # MLP sub-layer
        h = x  # Another residual connection
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

    def load_from(self, weights, n_block):
        """Load weights from pretrained model.

        Args:
            weights: Pretrained weights dictionary
            n_block: Block number to load weights for
        """
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # Load attention weights
            query_weight = np2th(weights[f"{ROOT}/{ATTENTION_Q}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[f"{ROOT}/{ATTENTION_K}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[f"{ROOT}/{ATTENTION_V}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[f"{ROOT}/{ATTENTION_OUT}/kernel"]).view(self.hidden_size, self.hidden_size).t()

            # Load attention biases
            query_bias = np2th(weights[f"{ROOT}/{ATTENTION_Q}/bias"]).view(-1)
            key_bias = np2th(weights[f"{ROOT}/{ATTENTION_K}/bias"]).view(-1)
            value_bias = np2th(weights[f"{ROOT}/{ATTENTION_V}/bias"]).view(-1)
            out_bias = np2th(weights[f"{ROOT}/{ATTENTION_OUT}/bias"]).view(-1)

            # Copy weights and biases
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # Load MLP weights
            mlp_weight_0 = np2th(weights[f"{ROOT}/{FC_0}/kernel"]).t()
            mlp_weight_1 = np2th(weights[f"{ROOT}/{FC_1}/kernel"]).t()
            mlp_bias_0 = np2th(weights[f"{ROOT}/{FC_0}/bias"]).t()
            mlp_bias_1 = np2th(weights[f"{ROOT}/{FC_1}/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            # Load normalization layer weights
            self.attention_norm.weight.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[f"{ROOT}/{ATTENTION_NORM}/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[f"{ROOT}/{MLP_NORM}/bias"]))


class Encoder(nn.Module):
    """Transformer encoder consisting of multiple transformer blocks."""

    def __init__(self, config, vis):
        """Initialize encoder.

        Args:
            config: Model configuration object
            vis (bool): Whether to return attention weights for visualization
        """
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # Create stack of transformer blocks
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        """Forward pass for encoder.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            tuple: (encoded_output, attention_weights_list)
        """
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)

        # Apply final layer normalization
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    """Complete Transformer model including embeddings and encoder."""

    def __init__(self, config, img_size, vis):
        """Initialize transformer.

        Args:
            config: Model configuration object
            img_size (int or tuple): Input image size
            vis (bool): Whether to return attention weights for visualization
        """
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        """Forward pass for transformer.

        Args:
            input_ids (torch.Tensor): Input image tensor of shape [batch, channels, height, width]

        Returns:
            tuple: (encoded_output, attention_weights)
        """
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model for image classification."""

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        """Initialize Vision Transformer.

        Args:
            config: Model configuration object
            img_size (int or tuple): Input image size (default: 224)
            num_classes (int): Number of output classes (default: 21843)
            zero_head (bool): Whether to initialize classification head with zeros (default: False)
            vis (bool): Whether to return attention weights for visualization (default: False)
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        # Transformer backbone
        self.transformer = Transformer(config, img_size, vis)

        # Classification head
        self.head = Linear(config.hidden_size, num_classes)

        # Initialize head weights
        if zero_head:
            nn.init.normal_(self.head.weight, std=0.2)  # Increased from 0.02
            nn.init.constant_(self.head.bias, 0.1)  # Non-zero initialization
        else:
            with np.load(args.pretrained_dir) as weights:
                self.head.weight.data.copy_(torch.from_numpy(weights["head/kernel"]).t() * 5)  # Amplify
                self.head.bias.data.copy_(torch.from_numpy(weights["head/bias"]) * 2)

        # LayerNorm before head
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, labels=None):
        """Forward pass for Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor
            labels (torch.Tensor, optional): Ground truth labels for loss computation

        Returns:
            torch.Tensor or tuple: If labels provided, returns loss. Otherwise returns (logits, attention_weights)
        """
        # Pass through transformer
        x, attn_weights = self.transformer(x)

        # Use class token for classification
        x = x[:, 0] * 2.0  # Scale class token features

        # Debug prints
        print(f"Pre-head features mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        # Classification head
        logits = self.head(x)
        print(f"Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        return logits, attn_weights

    def load_from(self, weights):
        """Load weights from pretrained model.

        Args:
            weights: Pretrained weights dictionary
        """
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            # Loading patch embeddings
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))

            # Loading class token
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            # Loading final layer norm
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            # Loading position embeddings (with possible interpolation for different sizes)
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                # Interpolating position embeddings for different grid sizes
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Loading transformer blocks
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # Loading hybrid model weights if using hybrid architecture
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),  # Base model with 16x16 patches
    'ViT-B_32': configs.get_b32_config(),  # Base model with 32x32 patches
    'ViT-L_16': configs.get_l16_config(),  # Large model with 16x16 patches
    'ViT-L_32': configs.get_l32_config(),  # Large model with 32x32 patches
    'ViT-H_14': configs.get_h14_config(),  # Huge model with 14x14 patches
    'R50-ViT-B_16': configs.get_r50_b16_config(),  # Hybrid ResNet-50 + ViT-B/16
    'testing': configs.get_testing(),  # Minimal config for testing
}