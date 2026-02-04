from setuptools import setup, find_packages

setup(
    name="py-anndata-visualizer",
    version="0.2.0-beta",
    author="Zachary Stensland",
    author_email="zach.stensland@ucsf.edu",
    description="Interactive spatial plotting for AnnData single-cell data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/py-anndata-visualizer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "py_anndata_visualizer": [
            "html/*.html",
            "js/*.js",
            "css/*.css",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "ipywidgets",
        "IPython",
        "scanpy",
        "anndata",
        "squidpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)