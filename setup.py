import os

from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="clip_interrogator",
    version="0.6.0",
    license='MIT',
    author='pharmapsychotic',
    author_email='me@pharmapsychotic.com',
    url='https://github.com/pharmapsychotic/clip-interrogator',
    description="Generate a prompt from an image",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    extras_require={'dev': ['pytest']},
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['blip','clip','prompt-engineering','stable-diffusion','text-to-image'],
)
