import numpy as np


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)








def env_state_shapes(env):
    return [obs.shape for obs in env.observation_space.spaces] \
        if hasattr(env,'spaces') else [env.observation_space.shape]
