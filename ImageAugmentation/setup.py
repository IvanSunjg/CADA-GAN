import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='augmentations',
    version='1.1.0',
    author='Jiugeng Sun, Jiaqing Xie, Sofie, Sofie Daniëls, Dan',
    author_email='jiugengsun@gmail.com',
    description='Data Augmentations for the Child Face GAN',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/IvanSunjg/ETH_DL_2022/tree/main/ImageAugmentation',
    license='MIT',
    packages=['augmentations'],
    install_requires=['numpy', 'matplotlib', 'torch', 'torchvision'],
)
