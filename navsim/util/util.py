import numpy as np


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def image_layout(x, old: str, new: str):
    new = [old.index(char) for char in new]
    return x.transpose(new)


def hwc_to_chw(s):
    if s.ndim == 3:
        s_ = image_layout(s, 'hwc', 'chw')
    elif s.ndim == 4:
        s_ = image_layout(s, 'nhwc', 'nchw')
    return s_


def s_hwc_to_chw(s):
    # TODO: HWC to CHW conversion optimized here
    # because pytorch can only deal with images in CHW format
    # we are making the optimization here to convert from HWC to CHW format.
    if isinstance(s, np.ndarray) and (s.ndim > 2):
        s = hwc_to_chw(s)
    elif isinstance(s, list):  # state is a list of states
        for i in range(len(s)):
            if isinstance(s[i], np.ndarray) and (s[i].ndim > 2):
                s[i] = hwc_to_chw(s[i])
    return s


def env_state_shapes(env):
    return [obs.shape for obs in env.observation_space.spaces] \
        if hasattr(env,'spaces') else [env.observation_space.shape]
