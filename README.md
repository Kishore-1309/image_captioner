# Image Captioning Project 🖼️✍️

Generate captions for images using a bridge between CLIP + GPT-2

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Kishore-1309/image_captioner.git
cd image_captioner

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

### Extract Features
```bash
python -m image_captioner.feature_extractor \
  --image_dir ./data/images \
  --save_dir ./features
```

### Train Model
```bash
python -m image_captioner.train \
  --features ./features \
  --captions ./data/captions.csv
```

### Generate Captions
```bash
python -m image_captioner.generate \
  --image test.jpg \
  --model checkpoints/best_model.pt
```

## 📂 Project Structure
```
project/
├── image_captioner/      # Core package
├── data/                 # Put your images here
├── features/             # CLIP features storage
└── checkpoints/          # Saved models
```

## 🤝 Contributing
1. Fork the project
2. Create your branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
MIT