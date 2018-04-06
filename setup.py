from setuptools import setup, find_packages

setup(name='TorchLanguage',
      version='0.1',
      description="A pyTorch toolkit for Natural Language Processing.",
      long_description="TorchLanguage is the equivalent of TorchVision for Natural Language Processing. It gives you access to text transformers (tokens, index, n-grams, etc), datasets, pre-trained models and embedding.",
      author='Nils Schaetti',
      author_email='nils.schaetti@unine.ch',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'torch',
          'numpy'
      ],
      zip_safe=False
      )

