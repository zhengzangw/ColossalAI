"""
need to install smooth quant https://github.com/mit-han-lab/smoothquant/tree/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c
Smooth Quant doesn't support generate, so we need to implement it by ourselves.
Its bmm only supports tensor with seq_len % 16 == 0, so we need to rewrite its forward,
we pad the inputs before bmm and unpad the results after bmm in attention and decode layer
"""

from typing import List, Optional, Tuple

import torch
from smoothquant.opt import (Int8OPTAttention, Int8OPTDecoder,
                             Int8OPTDecoderLayer, Int8OPTForCausalLM,
                             Int8OPTModel, LayerNormQ, W8A8B8O8LinearReLU,
                             W8A8BFP32OFP32Linear)
from torch import nn
from torch.nn.functional import pad


class GenInt8OPTForCausalLM(Int8OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = GenInt8OPTModel(config)

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = GenInt8OPTForCausalLM(module.config)
        int8_module.model = GenInt8OPTModel.from_float(
            module.model, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module


class GenInt8OPTModel(Int8OPTModel):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = GenInt8OPTDecoder(config)

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = GenInt8OPTModel(module.config)
        int8_module.decoder = GenInt8OPTModel.from_float(
            module.decoder, decoder_layer_scales)
        return int8_module


class GenInt8OPTDecoder(Int8OPTDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GenInt8OPTDecoderLayer(config.hidden_size, config.num_attention_heads, config.ffn_dim) for _ in range(config.num_hidden_layers)])

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = GenInt8OPTDecoder(module.config)
        int8_module.embed_tokens = module.embed_tokens
        int8_module.embed_positions = module.embed_positions
        int8_module.project_out = module.project_out
        int8_module.final_layer_norm = module.final_layer_norm
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = GenInt8OPTDecoderLayer.from_float(
                layer, **decoder_layer_scales[i])
        return int8_module

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = self.old_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        return output


class GenInt8OPTDecoderLayer(Int8OPTDecoderLayer):
    def __init__(self, embed_dim, num_attention_heads, ffn_dim):
        super().__init__(embed_dim, num_attention_heads, ffn_dim)
        self.self_attn = GenInt8OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads
        )

    @staticmethod
    def from_float(module,
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        int8_module = GenInt8OPTDecoderLayer(
            module.embed_dim,
            module.self_attn.num_heads,
            module.fc1.out_features
        )
        int8_module.self_attn_layer_norm = LayerNormQ.from_float(
            module.self_attn_layer_norm, attn_input_scale)
        int8_module.self_attn = GenInt8OPTAttention.from_float(
            module.self_attn, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)
        int8_module.final_layer_norm = LayerNormQ.from_float(
            module.final_layer_norm, fc1_input_scale)
        int8_module.fc1 = W8A8B8O8LinearReLU.from_float(
            module.fc1, fc1_input_scale, fc2_input_scale)
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(
            module.fc2, fc2_input_scale)
        return int8_module


class GenInt8OPTAttention(Int8OPTAttention):
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        # input len of bmm must be devided by 16
        pad_len_tgt = 16 - tgt_len % 16 if tgt_len % 16 != 0 else 0
        query_states = pad(query_states, (0, 0, 0, pad_len_tgt, 0, 0), value=0)
        pad_len_src = 16 - src_len % 16 if src_len % 16 != 0 else 0
        key_states = pad(key_states, (0, 0, 0, pad_len_src, 0, 0), value=0)

        attn_weights = self.qk_bmm(query_states, key_states)
        attn_weights = attn_weights[:, :tgt_len, :src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(
                1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        value_states = value_states.transpose(1, 2).contiguous()

        value_states = pad(value_states, (0, pad_len_src, 0, 0, 0, 0), value=0)
        attn_probs = pad(attn_probs, (0, pad_len_src,
                         0, pad_len_tgt, 0, 0), value=0)

        attn_output = self.pv_bmm(attn_probs, value_states)
        attn_output = attn_output[:, :tgt_len, :]

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(
            bsz, tgt_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value
