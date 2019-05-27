from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch_tensorflow_image_ml'))

VERSION = 0.1

setup(name='pytorch_tensorflow_image_ml',
      version=VERSION,
      description='Robot agent using an RL HTN',
      url='https://github.com/josiahls/PyTorchTensorflowImageML',
      author='Josiah Laivins',
      author_email='jlaivins@uncc.edu',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('pytorch_tensorflow_image_ml')],
      zip_safe=False,
      install_requires=['numpy', 'torch', 'tensorboardX', 'namedlist', 'pytest', 'torchvision', 'pandas'
                        ]
)
