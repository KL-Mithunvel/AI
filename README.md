# AI
Artificial Intelligence course – BMEE407L Term Paper This project is part of the a course (BMEE407L) at VIT Chennai, focused on applying AI techniques to real-world industrial systems. 
# 🧱 Tile Quality Inspection using AI

This repository is part of our BMEE407L Artificial Intelligence term project at VIT Chennai. Our goal is to build an AI-powered quality control system for tiles using computer vision techniques.

---

## 📁 Project Structure & What to Do in Each Folder

### `data/`
> Where all your data lives.

- `raw/`: Store unprocessed tile images here. You can create subfolders like `Type1/`, `Type2/`, etc.
- `processed/`: Store resized, cleaned, or augmented data here after running preprocessing scripts.
- `annotations/`: Label files for object detection or classification. Could be YOLO `.txt`, Pascal VOC `.xml`, or `.csv`.
- `samples/`: A few example images for testing code quickly.

✅ What to do: Collect tile images, label them, and organize them here.

---

### `notebooks/`
> Jupyter notebooks for development, training, and testing.

- `1_data_exploration.ipynb`: Visualize class distribution, check image sizes, etc.
- `2_preprocessing.ipynb`: Resize, normalize, and augment your images.
- `3_model_training.ipynb`: Train classification or detection model here.
- `4_inference_tests.ipynb`: Test trained model on new images.

✅ What to do: Use notebooks to explore and build your initial pipeline.

---

### `src/`
> Core source code (Python modules).

- `dataset.py`: Functions to load and preprocess image data.
- `model.py`: Define your CNN model here (e.g., MobileNet, ResNet).
- `train.py`: Script to train model using `dataset.py` and `model.py`.
- `evaluate.py`: Evaluate trained model, show confusion matrix or mAP.
- `infer.py`: Run inference on a folder or image.
- `utils.py`: Misc utilities like plotting, accuracy tracking, etc.

✅ What to do: Write clean, reusable code here for training and inference.

---

### `deployment/`
> For Raspberry Pi or real-time testing setup.

- `camera_capture.py`: Captures images from Pi Camera or webcam.
- `live_inference.py`: Runs model on live feed and shows prediction.
- `raspi_setup.md`: Instructions to run the project on Raspberry Pi.

✅ What to do: Use this when integrating with real hardware or testing in production.

---

### `reports/`
> Formal documents for submission.

- `DA_1.pdf`, `DA_2.pdf`, `DA_3.pdf`: Term paper deliverables.
- `turnitin_certificates/`: Plagiarism reports for each submission.

✅ What to do: Drop all final submission PDFs and certificates here.

---

### `demo/`
> Everything for your presentation/demo.

- `demo_script.md`: Script for your live or recorded demo.
- `test_video.mp4`: Screen-recorded or camera-based demo clip.
- `screenshots/`: Output images showing defect detection.

✅ What to do: Keep all demo materials organized here.

---

### Root Files

- `requirements.txt`: Python libraries required for the project.
- `README.md`: This file – an overview and guide for contributors.
- `.gitignore`: Tells Git which files/folders to ignore (like logs, weights).
- `LICENSE`: Optional – add an open-source license if needed.

---

## ✅ Getting Started

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/tile-quality-inspection-ai.git
cd tile-quality-inspection-ai

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run notebook to explore or train
jupyter notebook notebooks/1_data_exploration.ipynb
```

---

## 👥 Contributors

- Mithunvel K L 
- [Friend 1]
- [Friend 2]

---

Let's automate defect detection and make industrial quality control smarter! 🧠🔍🏭
