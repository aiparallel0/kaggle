"""
setup.py — Package configuration for DONUT+TrOCR+YOLO KIE Pipeline

This allows installation via:
    pip install .
    pip install -e .  (editable/development mode)

Or directly from GitHub:
    pip install git+https://github.com/aiparallel0/kaggle.git

After installation, run:
    python -m run_all
    python -m run_all -quick
    python -m run_all -quick -all
"""

from pathlib import Path

from setuptools import setup


# Read requirements
def read_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return []

    with open(req_file) as f:
        lines = f.readlines()

    # Filter out comments and blank lines
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="donut-kiedata",
    version="1.0.0",
    description="DONUT + TrOCR + YOLO multi-dataset KIE pipeline for receipt understanding",
    author="AI Parallel Team",
    author_email="contact@aiparallel.com",
    url="https://github.com/aiparallel0/kaggle",

    # Package configuration
    py_modules=["run_all"],  # Single module entry point
    install_requires=read_requirements(),

    # Entry point for CLI
    entry_points={
        "console_scripts": [
            "donut-kie=run_all:main",  # Install as CLI command: donut-kie
        ],
    },

    # Metadata
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords="donut transformers ocr kie receipts yolo trocr",
    long_description=Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8")
    if Path(__file__).parent.joinpath("README.md").exists() else "",
    long_description_content_type="text/markdown",
)
