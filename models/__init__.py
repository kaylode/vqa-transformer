from .transformer import Transformer
from .captioning import Captioning

def get_transformer_model(patch_size, trg_vocab):

    transformer_config = {
        'patch_size':       patch_size,
        'trg_vocab':        trg_vocab, 
        "d_model":          512, 
        "d_ff":             3072,
        "N_enc":            6,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3,
        "num_channels":     3,
        'pretrained_encoder': True
    }

    return Transformer(**transformer_config)