from .registry import BASE_MODEL, TOP_MODEL, FACE_MODEL

def _build(cfg, registry, default_args=None):
    assert(isinstance(cfg, dict) and 'type' in cfg)
    assert(isinstance(default_args, dict) or default_args is None)
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or a valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def build_model(cfg):
    return _build(cfg, FACE_MODEL)

def build_base_model(cfg):
    return _build(cfg, BASE_MODEL)

def build_top_model(cfg):
    return _build(cfg, TOP_MODEL)

