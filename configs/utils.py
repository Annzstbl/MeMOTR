# @Author       : Ruopeng Gao


import argparse


def update_config_with_kv(config: dict, k: str, v) -> [bool, dict]:
    """
    Update config with a pair of K and V from options.

    Args:
        config: Current config.
        k: A key from options.
        v: A value from options.

    Returns:
        [New config dict, Hit or Not]
    """
    hit = False
    for config_k in config.keys():
        if isinstance(config[config_k], dict):
            hit, config[config_k] = update_config_with_kv(config=config[config_k], k=k, v=v)
            if hit:
                break
        elif config_k == k.upper():
            if v == "True":
                config[config_k] = True
            elif v == "False":
                config[config_k] = False
            else:
                config[config_k] = v
            hit = True
            break
    return hit, config


def update_config(config: dict, option: argparse.Namespace) -> dict:
    """
    Update current config with an option parser.

    Args:
        config: Current config.
        option: Option parser.

    Returns:
        New config dict.
    """
    if is_unique(config)[0] is False:
        raise RuntimeError("Config's key is not unique, Please check the config file.")

    for option_k, option_v in vars(option).items():
        if option_k != "config_path" and option_v is not None:
            # except --config-path
            hit, config = update_config_with_kv(config=config, k=option_k, v=option_v)
            if hit is False:
                raise RuntimeError("The option '--%s' is not appeared in .yaml config file." % option_k)
    return config


def is_unique(config: dict) -> tuple[bool, set]:
    """
    Check whether keys are unique **within each mapping only**.

    Nested dicts may reuse the same key names as a parent (e.g. ``STAGE1.LR_DROP_RATE`` and
    top-level ``LR_DROP_RATE``); the previous implementation used one global ``keys_set`` for
    the whole tree and incorrectly rejected such configs.
    """
    level_keys = list(config.keys())
    if len(level_keys) != len(set(level_keys)):
        return False, set(level_keys)
    for v in config.values():
        if isinstance(v, dict):
            ok, dup = is_unique(v)
            if not ok:
                return False, dup
    return True, set()



