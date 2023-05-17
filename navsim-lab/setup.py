import setuptools

with open('../version.txt', 'r') as vf:
    __version__ = vf.read().strip()
# from navsim.version import __version__

# TODO: Replace the long description comments once cleared
#with open("../README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()
long_description = " "

setuptools.setup(
    name="navsim_lab",
    version=__version__,
    author="Armando Fandango, STTC, IST, UCF",
    author_email="armando@ucf.edu",
    description="Navigation Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    include_package_data=True,
    # data_files=[('navsim',['navsim/version.txt'])],
    python_requires='>=3.8',
    install_requires=[
        'navsim_envs',
        'cattrs',
        'scikit-fmm',
        'scikit-image',
        'tensorboard',
        'protobuf<=3.19',
        'tqdm',
        # 'opencv-python',
    ],
    entry_points={
        "console_scripts": [
            "navsim=navsim_lab.main:main",

        ]
    },
)
