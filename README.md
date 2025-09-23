# AI
Artificial Intelligence course – BMEE407L Term Paper This project is part of the a course (BMEE407L) at VIT Chennai, focused on applying AI techniques to real-world industrial systems. 
# 🧱 Tile Quality Inspection using AI

This repository is part of our BMEE407L Artificial Intelligence term project at VIT Chennai. Our goal is to build an AI-powered quality control system for tiles using computer vision techniques.

---

## 📁 Project Structure & What to Do in Each Folder

### `data/`
> Where all your data lives.

- `raw/`: Store unprocessed tile images here. You can create subfolders like `A/`, `B/`, etc.
- `processed/`: Store resized, cleaned, or augmented data here after running preprocessing scripts.
- `annotations/`: Label files for object detection or classification. Could be YOLO `.txt`, Pascal VOC `.xml`, or `.csv`.
- `samples/`: A few example images for testing code quickly.

---

### `notebooks/`
> Jupyter notebooks for development, training, and testing.

- `CNN1.ipynb`: Visualize class distribution, check image sizes, etc.

---

### `src/`
> Core source code (Python modules).

- `RawProcessingt.py`: Just to check the basic setting to check data.
- `crop.py`: Crop the images as needed.
- `prep_data.py`:  reformat the files as needed for robo studio.

---
<!--
### `deployment/`
> For Raspberry Pi or real-time testing setup.

- `camera_capture.py`: Captures images from Pi Camera or webcam.
- `live_inference.py`: Runs model on live feed and shows prediction.
- `raspi_setup.md`: Instructions to run the project on Raspberry Pi.

✅ What to do: Use this when integrating with real hardware or testing in production.
-->
---

### `reports/`
> Formal documents for submission.

- `DA_1.pdf`, `DA_2.pdf`, `DA_3.pdf`: Term paper deliverables.
- `turnitin_certificates/`: Plagiarism reports for each submission.

---
<!-- 
### `demo/`
> Everything for your presentation/demo.

- `demo_script.md`: Script for your live or recorded demo.
- `test_video.mp4`: Screen-recorded or camera-based demo clip.
- `screenshots/`: Output images showing defect detection.

✅ What to do: Keep all demo materials organized here.
-->
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
git clone https://github.com/KL-Mithunvel/AI.git
cd root diractory

# Step 2: Install dependencies
pip install -r requirements.txt

```

---

## 👥 Contributors

- Madumitha V
- Mithunvel K L 
- Kiran S
- subbu something 

---

Let's automate defect detection and make industrial quality control smarter! 🧠🔍🏭
