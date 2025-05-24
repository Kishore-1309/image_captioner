from setuptools import setup, find_packages

setup(
    name="image-captioner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.25.0",
        "pandas>=1.5.0",
        "tqdm>=4.64.0",
        "Pillow>=9.3.0"
        "clip @ git+https://github.com/openai/CLIP.git"
    ],
    entry_points={
        "console_scripts": [
            "captioner-train=image_captioner.train:train",
            "captioner-extract=image_captioner.feature_extractor:extract_features"
        ],
    },
)