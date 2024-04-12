from setuptools import setup

setup(
    name='custom_envs',
    version='0.0.1',
    packages=['custom_envs'],
    install_requires=['gymnasium'],
    description='Custom Environments for Reinforcement Learning',
    author='Bernd Frauenknecht, Emma Cramer, Ramil Sabirov, Lukas Kesper',
    options={'clean': {'all': True}}
)