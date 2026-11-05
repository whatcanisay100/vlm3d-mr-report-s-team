def _cfg(**kwargs):
    value = {
        "url": "",
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    }
    value.update(kwargs)
    return value
