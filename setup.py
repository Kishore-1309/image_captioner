from setuptools import setup, find_packages

setup(
    name="image-captioner",
    version="1.0.0",
    packages=find_packages(),
    description="Image Captioning with CLIP and GPT-2 using Cross-Attention",
    author="Chandra Kishore",
    author_email="chkishoreg@gmail.com",
    install_requires=[
        "torch>=2.6.0,<2.7.0",
        "transformers>=4.51.0,<4.52.0",
        "pandas>=2.2.0,<2.3.0",
        "tqdm>=4.67.0,<4.68.0",
        "Pillow>=11.1.0,<11.2.0",
        "matplotlib>=3.5,<4.0",
        "clip @ git+https://github.com/openai/CLIP.git"
    ],
    entry_points={
        "console_scripts": [
            "captioner-train=image_captioner.train:train",
            "captioner-preprocess=image_captioner.preprocess:main",
            "captioner-extract=image_captioner.feature_extractor:extract_features"
        ],
    },
)