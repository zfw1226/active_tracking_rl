from setuptools import setup
import sys
########################################################################################
#  Modified from https://github.com/zuoxingdong/gym-maze
########################################################################################
# Only support Python 3
if sys.version_info.major != 3:
    print('WARNING: This package only officially support Python 3, the current version is Python {sys.version_info.major}. The installation will likely fail. ')

setup(name='gym_track2d',
      install_requires=['gym', 
                        'numpy', 
                        'matplotlib', 
                        'scikit-image', ],
      description='gym-track2d: A customizable gym environment for active tracking in maze and grid world',
      version='0.1'
)
