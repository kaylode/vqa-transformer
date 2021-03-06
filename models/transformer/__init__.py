import copy
import torch.nn as nn
from .embedding import Embeddings, PositionalEncoding, PatchEmbedding, FeatureEmbedding, SpatialEncoding
from .layers import EncoderLayer, DecoderLayer
from .norm import LayerNorm
from .utils import draw_attention_map, init_xavier

TIMM_MODELS = [
        "deit_tiny_distilled_patch16_224", 
        'deit_small_distilled_patch16_224', 
        'deit_base_distilled_patch16_224',
        'deit_base_distilled_patch16_384']

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_pretrained_encoder(model_name):
    import timm
    assert model_name in TIMM_MODELS, "Timm Model not found"
    model = timm.create_model(model_name, pretrained=True)
    return model

class EncoderVIT(nn.Module):
    """
    Pretrained Transformers Encoder from timm Vision Transformers
    :output:
        encoded embeddings shape [batch * (image_size/patch_size)**2 * model_dim]
    """
    def __init__(self, model_name='deit_base_distilled_patch16_224'):
        super().__init__()
        
        vit = get_pretrained_encoder(model_name)
        self.embed_dim = vit.embed_dim 
        self.patch_embed = vit.patch_embed
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        
    def forward(self, src):
        x = self.patch_embed(src)
        x = self.pos_drop(x + self.pos_embed[:, 2:]) # skip dis+cls tokens
        x = self.blocks(x)
        x = self.norm(x)
        return x

class EncoderBottomUp(nn.Module):
    """
    Core encoder is a stack of N EncoderLayers
    :input:
        feat_dim:       feature dim
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, feat_dim, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.feat_embed = FeatureEmbedding(feat_dim, d_model)
        self.loc_embed = SpatialEncoding(d_model)
        if N > 0:
            self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
            self.norm = LayerNorm(d_model)    
    def forward(self, src, spatial_src):
        x = self.feat_embed(src)
        spatial_x = self.loc_embed(spatial_src)
        x += spatial_x

        if self.N > 0:
            for i in range(self.N):
                x = self.layers[i](x, mask=None)
            x = self.norm(x)
        return x

class Decoder(nn.Module):
    """
    Decoder with N-stacked DecoderLayers
    :input:
        vocab_size:     size of target vocab
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        decoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, vocab_size, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
        self.layers = get_clones(DecoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)
    def forward(self, trg, e_outputs):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, None, None)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Transformer model
    :input:
        patch_size:    size of patch
        trg_vocab:     size of target vocab
        d_model:       embeddings dim
        d_ff:          feed-forward dim
        N:             number of layers
        heads:         number of attetion heads
        dropout:       dropout rate
    :output:
        next words probability shape [batch * input length * vocab_dim]
    """
    def __init__(self, trg_vocab, num_classes, d_ff=3072, N_dec=4, heads=12, dropout=0.2):
        super().__init__()
        self.name = "Transformer"

        # Override decoder hidden dim if use pretrained encoder
        self.encoder = EncoderVIT()
        d_model = self.encoder.embed_dim

        self.decoder = Decoder(trg_vocab, d_model, d_ff, N_dec, heads, dropout)
        self.out = nn.Linear(d_model, num_classes)

        init_xavier(self.decoder)
        init_xavier(self.out)

    def forward(self, src, trg, src_mask, trg_mask, *args, **kwargs):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)

        # Aggregate
        output = self.out(d_output).mean(dim=1)
        return output
        
class TransformerBottomUp(nn.Module):
    """
    Transformer model
    :input:
        patch_size:    size of patch
        trg_vocab:     size of target vocab
        d_model:       embeddings dim
        d_ff:          feed-forward dim
        N:             number of layers
        heads:         number of attetion heads
        dropout:       dropout rate
    :output:
        next words probability shape [batch * input length * vocab_dim]
    """
    def __init__(self, trg_vocab, feat_dim, num_classes, d_model=768, d_ff=3072, N_enc=12, N_dec=4, heads=12, dropout=0.2):
        super().__init__()
        self.name = "TransformerBottomUp"

        self.encoder = EncoderBottomUp(feat_dim, d_model, d_ff, N_enc, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, d_ff, N_dec, heads, dropout)
        self.out = nn.Linear(d_model, num_classes)
        init_xavier(self)

    def forward(self, src, loc_src, trg, *args, **kwargs):
        e_outputs = self.encoder(src, loc_src)
        d_output = self.decoder(trg, e_outputs)

        # Aggregate
        output = self.out(d_output).mean(dim=1)
        return output