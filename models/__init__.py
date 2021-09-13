from .transformer import Transformer, TransformerBottomUp
from .captioning import Captioning

def get_transformer_model(trg_vocab, num_classes):

    transformer_config = {
        'trg_vocab':        trg_vocab, 
        'num_classes':      num_classes,
        "d_ff":             3072,
        "N_dec":            6,
        "heads":            8,
        "dropout":          0.3,
    }

    return Transformer(**transformer_config)

def get_transformer_bottomup_model(bottom_up_dim, trg_vocab, num_classes):

    transformer_config = {
        'feat_dim':       bottom_up_dim,
        'trg_vocab':        trg_vocab, 
        'num_classes':      num_classes,
        "d_model":          512, 
        "d_ff":             2048,
        "N_enc":            3,
        "N_dec":            3,
        "heads":            4,
        "dropout":          0.1,
    }

    return TransformerBottomUp(**transformer_config)