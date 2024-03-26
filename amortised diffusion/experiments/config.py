import ml_collections

def D(config_dict):
    return ml_collections.ConfigDict(config_dict)


_PRETRAINED_WEIGHTS = {
    "mnist": {
        "amortized": {
            # The amortized network weights depend on the likelihood
            "inpainting": "weights/mnist_ddpm_unconditional.pth",
            "outpainting": "weights/mnist_ddpm_unconditional.pth",
        },
        "reconstruction_guidance": "weights/mnist_ddpm_unconditional.pth",
        "replacement": "weights/mnist_ddpm_unconditional.pth",
    },
    "flowers": {
        "amortized": {
            # The amortized network weights depend on the likelihood
            "inpainting": "weights/flowers_inpainting_amortized.pth",
            "outpainting": "weights/flowers_outpainting_amortized.pth",
        },
        "reconstruction_guidance": "weights/flowers_ddpm_unconditional.pth",
        "replacement": "weights/flowers_ddpm_unconditional.pth",
    },
    "celeba": {
        "amortized": {
            # The amortized network weights depend on the likelihood
            "inpainting": "weights/celeba_inpainting_amortized.pth",
            "outpainting": "weights/celeba_outpainting_amortized.pth",
        },
        "reconstruction_guidance": "weights/celeba_ddpm_unconditional.pth",
        "replacement": "weights/celeba_ddpm_unconditional.pth",
    },
}


_LIKELIHOOD_CONFIGS = {
    "inpainting": D({
        "name": "inpainting",
        "patch_size": 20,
        "pad_value": -2,
    }),
    "outpainting": D({
        "name": "outpainting",
        "patch_size": 24,
        "pad_value": -2,
    }),
    "hyperresolution": D({  
        "name": "hyperresolution",
        "target_height": 16,  
        "target_width": 16,  
    }),
}

_DATASET_CONFIGS = {
    "mnist": D({
        "name": "mnist",
        "image_size": 28,
        "num_channels": 1,
    }),
    "flowers": D({
        "name": "flowers",
        "image_size": 64,
        "num_channels": 3,
    }),
    "celeba": D({
        "name": "celeba",
        "image_size": 64,
        "num_channels": 3,
    }),
}


_CONDITIONING_CONFIGS = {
    "reconstruction_guidance": D({
        "name": "reconstruction_guidance",
        "gamma": 10.0,
        "start_fraction": 1.0,
        "update_rule": "before",
        "n_corrector": 0,
        "delta": 0.1,
    }),
    "replacement": D({
        "name": "replacement",
        "start_fraction": 1.0,
        "noise": True,
        "n_corrector": 0,
        "delta": 0.1,
    }),
    "amortized": D({
        "name": "amortized",
        "p_cond": 0.9,
        "n_corrector": 0,
        "delta": 0.1,
    }),
}


_NETWORK_CONFIGS = {
    "mnist": D({
        "num_channels": 32,
        "num_res_blocks": 1,
        "channel_mult": "1, 2, 2",
        "resblock_updown": True,
        # "model_path": "weights/mnist_ddpm_unconditional.pth",
    }), 
    "flowers": D({
        "num_channels": 128,
        "num_res_blocks": 1,
        "resblock_updown": True,
        "num_head_channels": 64,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
        "num_heads": 4,
    }), 
    "celeba": D({
        "num_channels": 128,
        "num_res_blocks": 1,
        "resblock_updown": True,
        "num_head_channels": 64,
        "resblock_updown": True,
        "use_scale_shift_norm": True,
        "num_heads": 4,
    }), 
}


def get_config(config_str):
    """
    Get the default hyperparameter configuration.

    config_str: dataset,likelihood,conditioning
    """
    if len(config_str.split(',')) != 3:
        print("config.py:<dataset>,<likelihood>,<conditioning>")
        exit(0)

    dataset, likelihood, conditioning = config_str.split(',')
    print(config_str.split(','))
    assert likelihood in _LIKELIHOOD_CONFIGS.keys(), (
        f"Unknown likelihood: {likelihood}."
        f"Valid options: {', '.join(list(_LIKELIHOOD_CONFIGS.keys()))}."
    )
    assert dataset in _DATASET_CONFIGS.keys(), (
        f"Unknown dataset: {dataset}."
        f"Valid options: {', '.join(list(_DATASET_CONFIGS.keys()))}."
    )
    assert conditioning in _CONDITIONING_CONFIGS.keys(), (
        f"Unknown conditioning: {conditioning}."
        f"Valid options: {', '.join(list(_CONDITIONING_CONFIGS.keys()))}."
    )

    config = ml_collections.ConfigDict()

    config.dataset = _DATASET_CONFIGS[dataset]
    config.network = _NETWORK_CONFIGS[dataset]

    try:
        if conditioning == "amortized":
            # The amortized network weights depend on the likelihood
            config.network.model_path = _PRETRAINED_WEIGHTS[dataset]["amortized"][likelihood]
        else:
            config.network.model_path = _PRETRAINED_WEIGHTS[dataset][conditioning]
    except:
        print("No pre-trained network for dataset...")
        config.network.model_path = ""

    config.likelihood = _LIKELIHOOD_CONFIGS[likelihood]
    config.conditioning = _CONDITIONING_CONFIGS[conditioning]

    config.training = ml_collections.ConfigDict({
        "num_epochs": 100 if dataset == "flowers" else 10,  # flowers is a very small dataset and requires many passes.
        "batch_size": 32,
        "lr_schedule": "constant",
        "lr_end_warmup": 1e-3,
        "lr_final": 1e-5,
        "warmup_steps": 1000,
    })
    assert config.training.lr_schedule in ["constant", "warmup_cosine"]

    config.diffusion = ml_collections.ConfigDict({
        "num_steps": 1000
    })

    config.testing = ml_collections.ConfigDict({
        "fid": False,
        "num_test": 96,
        # "num_test": 2048,
        "batch_size": 32,
        "seed": 0,
    })
    return config
