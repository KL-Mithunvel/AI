# AI
Artificial Intelligence course – BMEE407L Term Paper This project is part of the a course (BMEE407L) at VIT Chennai, focused on applying AI techniques to real-world industrial systems. 
# 🧱 Tile Quality Inspection using AI

This repository is part of our BMEE407L Artificial Intelligence term project at VIT Chennai. Our goal is to build an AI-powered quality control system for tiles using computer vision techniques.

> **Archived course prototype.** This is the older BMEE407L term-paper project.
> It does **not** match the requirements of the newer
> [`Tile_Sorting`](https://github.com/KL-Mithunvel/tile_sorting) automation
> system, but the methods here (camera capture/crop pipeline, classical tile
> segmentation, and the CNN grade classifier) are still reusable as a starting
> point. See the detailed write-ups below.

---

## 📚 Detailed documentation

Full, code-accurate documentation of what was built and how it works:

- **[`docs/METHODS.md`](docs/METHODS.md)** — the two tile-grading ("sorting")
  methods developed here: the classical CV checkerboard/tile segmentation
  (`RawProcessing.py`) and the CNN grade classifier (`CNN1.ipynb`), plus the
  end-to-end data flow and an honest read of the results.
- **[`docs/CAMERA_MODE.md`](docs/CAMERA_MODE.md)** — the interactive **camera
  mode**: perspective-crop + 4-rotation augmentation tool (`crop.py`), its
  operator workflow, image processing, and how it feeds the models.

---

## 📁 Actual repository structure

```text
AI/
├── CNN1.ipynb               # Colab notebook: preprocessing + 4-class CNN grade classifier
├── DATA PROCESSING/
│   ├── RawProcessing.py     # Classical CV: checkerboard + tile segmentation (no ML)
│   ├── crop.py              # "Camera mode": interactive perspective crop + rotation augmentation
│   └── prep_data.py         # Rename/flatten processed images for Roboflow ("robo studio")
├── DATASET/
│   └── RAW DATA/            # Grade-labelled raw photos: G3A/, G3B/, G4/, G5/ (the 4 classes)
├── DAs/                     # Digital Assignment / term-paper deliverables + reference paper
├── docs/                    # Detailed documentation (see above)
├── README.md
└── LICENSE
```

> Only `DATASET/RAW DATA/` (the four G-grade folders) is tracked in git. The
> working folders the scripts reference — `DATASET/Camera/…` (rig camera, used by
> `crop.py`) and `DATASET/Phone/…` (phone photos, used by `prep_data.py`) — plus
> the Drive-hosted `PROCESSED_DATA/` the notebook trains on, are local/Drive data
> and are intentionally not committed.

---

## ✅ Getting Started

```bash
# Step 1: Clone the repo
git clone https://github.com/KL-Mithunvel/AI.git
cd AI

# Step 2: Install the libraries the scripts need (there is no requirements.txt
# in this archived repo — the data-processing scripts use OpenCV + NumPy):
pip install opencv-python numpy
```

- **`DATA PROCESSING/crop.py` / `RawProcessing.py`** run locally and need OpenCV
  with GUI support (they open `cv2.imshow` windows). See
  [`docs/CAMERA_MODE.md`](docs/CAMERA_MODE.md) for the camera-mode workflow.
- **`CNN1.ipynb`** is a Google Colab notebook — open it in Colab (it mounts
  Google Drive for the dataset and uses TensorFlow). See
  [`docs/METHODS.md`](docs/METHODS.md) for the model details.

---

## 👥 Contributors

- Madumitha V
- Mithunvel K L 
- Kiran S
- subbu something 

---

Let's automate defect detection and make industrial quality control smarter! 🧠🔍🏭
