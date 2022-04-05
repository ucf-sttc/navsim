try:
    from cv2 import imwrite as imwrite

    print("navsim_envs: using cv2 as image library")
except ImportError as error:
    try:
        from imageio import imwrite as imwrite

        print("navsim_envs: using imageio as image library")
    except ImportError as error:
        try:
            from matplotlib.pyplot import imsave as imwrite

            print("navsim_envs: using matplotlib as image library")
        except ImportError as error:
            try:
                from PIL import Image

                print("navsim_envs: using PIL as image library")


                def imwrite(filename, arr):
                    im = Image.fromarray(arr)
                    im.save(filename)
            except ImportError as error:
                def imwrite(filename=None, arr=None):
                    print("navsim_envs: unable to load any of the following "
                          "image libraries: cv2, imageio, matplotlib, "
                          "PIL. Install one of these libraries to "
                          "save visuals.")


                imwrite()

def get_logger(name:str='navsim'):
    #import logging
    #logger = logging.getLogger('navsim')
    from mlagents_envs.logging_util import get_logger
    logger = get_logger(name)
    return logger

logger = get_logger()