# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import TransformerConfig

from fairseq.models.transformer import (
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
    TransformerDecoderCascade,
    TransformerEncoderCascade,
    TransformerModel,
)


@register_model("transformer_cascade")
class TransformerModelCascade(TransformerModel):

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerModelCascade, TransformerModelCascade).add_args(parser)
        parser.add_argument('--encoder-self-att-heads', type=str,
                            help='Numbers of self attention heads for each layer of encoder splitted by comma.' +
                            'If all layers have equal number of heads, only one number can be specified')
        parser.add_argument('--decoder-self-att-heads', type=str,
                            help='Numbers of self attention heads for each layer of decoder splitted by comma.' +
                            'If all layers have equal number of heads, only one number can be specified')
        parser.add_argument('--decoder-cross-att-heads', type=str,
                            help='Numbers of cross attention heads for each layer of decoder splitted by comma.' +
                            'If all layers have equal number of heads, only one number can be specified')

        parser.add_argument('--encoder-head-dim', type=int,
                            help='Dimension of heads in encoder self attention')
        parser.add_argument('--decoder-head-dim', type=int,
                            help='Dimension of heads in decoder self and cross attention')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        cfg = TransformerConfig.from_namespace(args)
        return TransformerEncoderCascade(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        cfg = TransformerConfig.from_namespace(args)
        return TransformerDecoderCascade(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


# @register_model_architecture("transformer_cascade", "transformer_cascade")
# def transformer_cascade(args):
#     args.encoder_head_dim = getattr(args, "encoder_head_dim", 128)
#     args.decoder_head_dim = getattr(args, "decoder_head_dim", 128)
#     transformer_iwslt_de_en(args)


@register_model_architecture("transformer_cascade", "transformer_cascade_wmt_en_fr_big")
def transformer_cascade_wmt_en_fr_big(args):
    args.encoder_self_att_heads = getattr(args, "encoder_self_att_heads", "8,8,16,16,24,24")
    args.decoder_self_att_heads = getattr(args, "decoder_self_att_heads", "8,8,16,16,24,24")
    args.decoder_cross_att_heads = getattr(args, "decoder_cross_att_heads", "8,8,16,16,24,24")
    args.encoder_head_dim = getattr(args, "encoder_head_dim", 64)
    args.decoder_head_dim = getattr(args, "decoder_head_dim", 64)
    transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture("transformer_cascade", "transformer_cascade_wmt_en_de_big")
def transformer_cascade_wmt_en_de_big(args):
    args.encoder_self_att_heads = getattr(args, "encoder_self_att_heads", "8,8,16,16,24,24")
    args.decoder_self_att_heads = getattr(args, "decoder_self_att_heads", "8,8,16,16,24,24")
    args.decoder_cross_att_heads = getattr(args, "decoder_cross_att_heads", "8,8,16,16,24,24")
    args.encoder_head_dim = getattr(args, "encoder_head_dim", 64)
    args.decoder_head_dim = getattr(args, "decoder_head_dim", 64)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_cascade", "transformer_cascade_iwslt_de_en")
def transformer_cascade_iwslt_de_en(args):
    args.encoder_self_att_heads = getattr(args, "encoder_self_att_heads", "4,4,4,4,4,4")
    args.decoder_self_att_heads = getattr(args, "decoder_self_att_heads", "4,4,4,4,4,4")
    args.decoder_cross_att_heads = getattr(args, "decoder_cross_att_heads", "2,2,4,4,6,6")
    args.encoder_head_dim = getattr(args, "encoder_head_dim", 128)
    args.decoder_head_dim = getattr(args, "decoder_head_dim", 128)
    transformer_iwslt_de_en(args)
