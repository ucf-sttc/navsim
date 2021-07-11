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


def s_hwc_to_chw(s):
    # TODO: HWC to CHW conversion optimized here
    # because pytorch can only deal with images in CHW format
    # we are making the optimization here to convert from HWC to CHW format.
    if isinstance(s, np.ndarray) and (s.ndim > 2):
        s = image_layout(s, 'hwc', 'chw')
    elif isinstance(s, list):  # state is a list of states
        for i in range(len(s)):
            if isinstance(s[i], np.ndarray):
                if (s[i].ndim == 3):
                    s[i] = image_layout(s[i], 'hwc', 'chw')
                elif (s[i].ndim == 4):
                    s[i] = image_layout(s[i], 'nhwc', 'nchw')

    return s
