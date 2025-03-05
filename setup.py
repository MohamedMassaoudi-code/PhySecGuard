from setuptools import setup, find_packages

setup(
    name='cyberphyslib',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python library for evaluating adversarial vulnerabilities in cyber-physical systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/cyberphyslib',  # Update with your repo URL
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
