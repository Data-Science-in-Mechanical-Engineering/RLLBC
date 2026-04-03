from setuptools import setup, find_packages

setup(
    name="custom_envs",
    version="0.0.1",
    packages=find_packages(include=["custom_envs", "custom_envs.*"]),
    include_package_data=True,
    package_data={"custom_envs": ["envs/img/*", "envs/font/*"]},
    install_requires=["gymnasium"],
    description="Custom Environments for Reinforcement Learning",
    author="Bernd Frauenknecht, Emma Cramer, Ramil Sabirov, Lukas Kesper",
    options={"clean": {"all": True}},
)