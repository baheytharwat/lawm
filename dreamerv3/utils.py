import re
import torch

class Config:
    """Config class that supports both attribute access and dictionary unpacking"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def keys(self):
        return vars(self).keys()
    
    def items(self):
        for key, value in vars(self).items():
            if isinstance(value, Config):
                yield key, dict(value.items())
            else:
                yield key, value
    
    def to_dict(self):
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def apply_size_override(config, size_key):
    """Apply overrides like 'size12m' to base config."""
    if size_key in config:
        overrides = config[size_key]
        for pattern, values in overrides.items():
            for conf_key in list(config['defaults'].keys()):
                if re.fullmatch(pattern, conf_key):
                    if isinstance(config['defaults'][conf_key], dict) and isinstance(values, dict):
                        config['defaults'][conf_key].update(values)
                    else:
                        config['defaults'][conf_key] = values
            
            if pattern not in config['defaults']:
                config['defaults'][pattern] = values

def setup_config_for_worldmodel(cfg_dict, size_overrides):
    """Setup configuration for WorldModel with proper heads for image-only training"""
    rssm_key = '.*\\.rssm'
    depth_key = '.*\\.depth'
    units_key = '.*\\.units'
    
    for pattern, values in size_overrides.items():
        if pattern == rssm_key:
            cfg_dict.update({
                'dyn_stoch': values.get('classes', 32),
                'dyn_deter': values.get('deter', 512), 
                'dyn_hidden': values.get('hidden', 512),
                'dyn_discrete': values.get('discrete', 32),
            })
        elif pattern == depth_key:
            cfg_dict['encoder'] = cfg_dict.get('encoder', {})
            cfg_dict['decoder'] = cfg_dict.get('decoder', {})
            cfg_dict['encoder']['cnn_depth'] = values
            cfg_dict['decoder']['cnn_depth'] = values
        elif pattern == units_key:
            cfg_dict['units'] = values
            cfg_dict['encoder'] = cfg_dict.get('encoder', {})
            cfg_dict['decoder'] = cfg_dict.get('decoder', {})
            cfg_dict['encoder']['mlp_units'] = values
            cfg_dict['decoder']['mlp_units'] = values
    
    required_defaults = {
        'dyn_rec_depth': 1,
        'dyn_mean_act': 'none',
        'dyn_std_act': 'sigmoid2',
        'dyn_min_std': 0.1,
        'unimix_ratio': 0.01,
        'initial': 'learned',
        'kl_free': 1.0,
        'dyn_scale': 0.5,
        'rep_scale': 0.1,
        'act': 'SiLU',
        'norm': True,
        'grad_heads': ['decoder'],  # Only train decoder, disable other heads
        
        # Training specific parameters
        'model_lr': 1e-4,
        'opt_eps': 1e-8,
        'grad_clip': 1000.0,
        'weight_decay': 0.0,
        'opt': 'adam',
        'precision': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        'encoder': {
            'mlp_keys': '$^',  # No MLP inputs - this regex matches nothing
            'cnn_keys': 'image',
            'act': 'SiLU',
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 1024,
            'symlog_inputs': True
        },
        
        'decoder': {
            'mlp_keys': '$^', 
            'cnn_keys': 'image',
            'act': 'SiLU', 
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 1024,
            'cnn_sigmoid': False,
            'image_dist': 'mse', 
            'vector_dist': 'symlog_mse',
            'outscale': 1.0
        },
    }

    for key, value in required_defaults.items():
        if key not in cfg_dict:
            cfg_dict[key] = value
        elif key in ['encoder', 'decoder'] and isinstance(cfg_dict[key], dict):
            for subkey, subvalue in value.items():
                if subkey not in cfg_dict[key]:
                    cfg_dict[key][subkey] = subvalue
    
    return cfg_dict