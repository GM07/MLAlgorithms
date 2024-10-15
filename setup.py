from setuptools import setup, find_packages

setup(
    name="mlalgorithms",
    version="0.1.0",
    author="Gaya Mehenni",
    author_email="mehgaya@gmail.com",
    description="A collection of machine learning algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gm07/mlalgorithms",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # Add your project dependencies here, for example:
        # 'numpy>=1.18.0',
        # 'scikit-learn>=0.22.0',
        'torch>=2.4.0'
    ],
)
