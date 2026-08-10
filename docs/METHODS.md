# Tile Sorting / Grading Methods — What This Project Actually Does

> **Status: archived course prototype.** This repository is the BMEE407L
> Artificial Intelligence term-paper project from VIT Chennai ("Tile Quality
> Inspection using AI"). It predates the current
> [`Tile_Sorting`](https://github.com/KL-Mithunvel/tile_sorting) automation
> project and does **not** match that system's requirements. It is kept because
> the methods below — the image-capture/preprocessing pipeline and the two
> grading approaches — are still reusable as a starting point and as a record of
> what was tried and what the results were.

This document explains, in detail, the sorting/grading methods that were
developed here through the models and scripts in the repo. "Sorting" in this
project means **classifying a tile into a quality grade** from a photograph — it
is a vision/classification problem, not a mechanical sorting mechanism. The
physical handling of tiles is out of scope here; that belongs to the newer
`Tile_Sorting` repo.

---

## 1. The grading target (classes)

The dataset is organised by **tile grade**, one folder per class:

| Class folder (`DATASET/RAW DATA/`) | Raw image count |
|---|---|
| `G3A` | 20 |
| `G3B` | 20 |
| `G4`  | 21 |
| `G5`  | 21 |

These four grade folders are what every model in the project is trained/measured
against — the CNN's final layer is a **4-way softmax**, one output per grade
class. The grade labels (`G3A`, `G3B`, `G4`, `G5`) come from the way the tiles
were bucketed when the photos were taken; this repo treats them purely as
class labels and does not encode any ordering or defect taxonomy beyond the
folder name.

The committed raw set is small (~82 photos). The models below are trained on a
larger **processed** set (328 images — see §4) that is produced from the raw
photos by the capture/augmentation pipeline and is stored on Google Drive, not
in git (image data is intentionally kept out of the repo).

---

## 2. Two grading methods were developed

The project explored two independent routes from a tile photo to a grade:

1. **Method A — Classical computer-vision segmentation** (`DATA PROCESSING/RawProcessing.py`):
   no machine learning. Locates the calibration checkerboard and the tile in the
   frame using gradient-energy texture analysis, morphology, and colour
   clustering. This is the "isolate the tile so we can measure/inspect it" stage.
2. **Method B — CNN grade classifier** (`CNN1.ipynb`): a convolutional neural
   network that takes a cropped, contrast-enhanced grayscale tile image and
   predicts one of the four grade classes directly.

Method A is deterministic and explainable but only *finds* the tile; Method B
*grades* it but is a black box. In the intended flow, A (or the manual camera
mode in §5) produces a clean, cropped tile image, and B classifies it.

---

## 3. Method A — Classical CV tile/checkerboard segmentation

File: `DATA PROCESSING/RawProcessing.py`

The capture rig places each tile next to a **checkerboard** (used as a size /
colour reference). The script separates the two.

### `highlight_checkerboard(img_bgr)`
Finds the checkerboard frame using the fact that a checkerboard is extremely
**edge-dense**:

1. **Texture energy** — convert to gray, take Scharr gradients in x and y,
   compute gradient magnitude, then a box-filtered **local energy** map (squared
   gradient averaged over an odd window ~1% of the image's smaller dimension).
   Normalised to 0–255.
2. **Threshold + morphology** — Otsu threshold the energy map, then
   `MORPH_CLOSE` (15×15) followed by `MORPH_OPEN` (5×5) to consolidate the
   pattern into a solid blob.
3. **Component selection** — `connectedComponentsWithStats`; keep only
   components that are both large (> 0.1% of image area) **and touch the image
   border** (the checkerboard frame runs to the edge of the staged shot). Clean
   with a `MORPH_CLOSE` (31×31) + dilate.
4. **Overlay** — fit a `minAreaRect` to the kept pixels and draw a translucent
   green quadrilateral. Returns `(overlay, checker_mask, energy_map)`.

### `highlight_tile(img_bgr, checker_mask, k=3)`
Finds the tile (the central terracotta patch), explicitly **excluding** the
checkerboard region:

1. Dilate `checker_mask` (21×21) to get a safe exclusion zone.
2. **K-means colour clustering** in **Lab** colour space (`k=3`) — Lab separates
   colour from luminance well, which suits terracotta-vs-background.
3. For each cluster: mask out the checkerboard zone, clean with
   `MORPH_OPEN` (7×7) + `MORPH_CLOSE` (31×31), drop small and border-touching
   components (the tile sits in the interior, unlike the checkerboard).
4. **Score** each candidate blob by `area × solidity`, where
   `solidity = contour_area / minAreaRect_area` (≈1 means a filled,
   rectangle-like region). The highest-scoring blob is the tile.
5. **Overlay** — draw a translucent blue quadrilateral over the winning tile.
   Returns `(overlay, tile_mask)`.

### `highlight_checkerboard_and_tile(img_bgr)`
Convenience wrapper: runs the checkerboard detector, feeds its mask into the tile
detector, returns the combined overlay plus both masks. The `__main__` block runs
it on a sample image and writes `highlighted_both.png`, `checker_mask.png`, and
`tile_mask.png`.

**What it gives the grading pipeline:** a `tile_mask` / cropped tile region with
the reference pattern removed — i.e. a clean input for either measurement or the
CNN. It does **not** itself decide a grade.

---

## 4. Method B — CNN grade classifier (`CNN1.ipynb`)

A Google Colab notebook that trains a 4-class CNN. Pipeline, cell by cell:

### 4.1 Crack-oriented preprocessing
Every processed image is turned into a **grayscale, crack-enhanced** image before
training (`ProcessedTiles_Grey/`):

1. `cvtColor → GRAY`.
2. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation, `clipLimit=3.0`,
   `tileGridSize=(8,8)`) to boost local contrast so surface defects stand out.
3. **High-pass / unsharp sharpen**: `addWeighted(enhanced, 1.5, blurred, -0.5, 0)`
   with a 5×5 Gaussian blur — emphasises thin cracks against the tile body.

The rationale is that grade differences are dominated by cracks/surface damage,
so the network is fed a representation that exaggerates exactly those features.

### 4.2 Dataset loading
- `image_dataset_from_directory` on `ProcessedTiles_Grey/`, `color_mode='grayscale'`,
  `image_size=(256,256)`, `batch_size=32`.
- Reports **"Found 328 files belonging to 4 classes."**
- Pixels normalised to `[0,1]` (`x/255`).
- Split **70 / 20 / 10** into train / val / test via `take`/`skip`.

### 4.3 Model architecture
`Sequential`, ~**18.35 M** parameters:

| Stage | Layers |
|---|---|
| Block 1 | Conv2D(32, 3×3, ReLU, same) → BatchNorm → MaxPool 2×2 |
| Block 2 | Conv2D(64) → BatchNorm → MaxPool |
| Block 3 | Conv2D(128) → BatchNorm → MaxPool |
| Block 4 | Conv2D(256) → BatchNorm → MaxPool |
| Block 5 | Conv2D(512) → BatchNorm → MaxPool |
| Head | Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(4, softmax) |

Input shape `(256, 256, 1)`. Compiled with **Adam**,
`SparseCategoricalCrossentropy`, metric `accuracy`. Trained for **50 epochs**
with a TensorBoard callback.

### 4.4 Results (and the honest read on them)
The recorded training run shows the model **overfitting badly**:

- Training accuracy climbs from ~0.69 to ~0.86–0.89 over 50 epochs.
- **Validation accuracy stays low and unstable** (~0.16–0.47, mostly ~0.25–0.30)
  and **validation loss is enormous** (≈100–195) throughout.

Interpretation: with only ~328 images across 4 classes and an 18 M-parameter
network, the model memorises the training set and does not generalise. The gap
between train and val is the classic small-data / over-capacity signature. This
result is *why* the project is considered a prototype rather than a deliverable
model — see "Reusability & next steps" below.

---

## 5. Data pipeline supporting the models

Two scripts prepare data for the models above. The interactive one is the
"camera mode" and is documented in full detail in
[`CAMERA_MODE.md`](CAMERA_MODE.md); summarised here for the end-to-end picture:

- **`DATA PROCESSING/crop.py` — camera mode (interactive perspective crop + rotation augmentation).**
  Walks `DATASET/Camera/RAWDATA`, lets the operator click and drag the tile's 4
  corners, warps the tile to a flat rectangle (`getPerspectiveTransform` +
  `warpPerspective`), and saves it at **4 rotations (0/90/180/270°)** into
  `DATASET/Camera/PROCESSED DATA` — a 4× data-augmentation from every hand-cropped
  photo.
- **`DATA PROCESSING/prep_data.py` — Roboflow prep.** Walks
  `DATASET/Phone/PROCESSED DATA`, keeps only image sets that have all four
  rotations, and renames/copies them to `{class}_pic{n}_r{angle}.jpg` under
  `final/` — a flat, consistently-named set ready to upload to Roboflow
  ("robo studio").
- **`DATA PROCESSING/RawProcessing.py`** — the Method A segmentation described
  above; used to check/visualise tile isolation on raw shots.

Note the two capture sources referenced in the scripts: **`DATASET/Camera/…`**
(the rig camera, handled by `crop.py`) and **`DATASET/Phone/…`** (phone photos,
handled by `prep_data.py`). Only `DATASET/RAW DATA/` (the G-grade folders) is
committed to git; the `Camera/` and `Phone/` working folders are local/Drive
data and are not tracked.

---

## 6. End-to-end flow

```text
Raw tile photo (rig camera or phone, tile beside a checkerboard)
        │
        ├─ Method A: RawProcessing.py ── highlight checkerboard + tile ─┐
        │   (automatic tile isolation, no ML)                          │
        │                                                              ▼
        └─ Camera mode: crop.py ── manual 4-corner perspective crop ── clean, flat tile image
                    │                     + 4 rotations (augmentation)
                    ▼
            prep_data.py ── rename/flatten for Roboflow
                    │
                    ▼
        CNN1.ipynb ── CLAHE + high-pass → grayscale 256×256
                    ── 5-block CNN → 4-class softmax
                    ▼
             Predicted grade: G3A / G3B / G4 / G5
```

---

## 7. Reusability & next steps (why it's kept)

What is directly reusable in the current `Tile_Sorting` project:

- **The camera-mode crop tool** (`crop.py`) — a solid, hand-driven way to build a
  clean, perspective-corrected, rotation-augmented tile dataset. See
  [`CAMERA_MODE.md`](CAMERA_MODE.md).
- **The classical segmentation** (`RawProcessing.py`) — checkerboard-aware tile
  isolation that could seed the newer camera node's automatic tile-region finder.
- **The crack-enhancement preprocessing** (CLAHE + unsharp) — a cheap way to make
  cracks salient, reusable regardless of the classifier that follows.

What should **not** be carried over as-is:

- **The CNN as trained** — it overfits on ~328 images. Any real use needs far more
  data (the camera-mode augmentation helps but is not enough on its own),
  regularisation/augmentation during training, a smaller or pretrained backbone
  (transfer learning), and a proper held-out test evaluation rather than the
  train-vs-val gap seen here.
- **The grade labels** — `G3A/G3B/G4/G5` are this dataset's buckets, not the
  grading scheme the production system settles on.

## 8. Reference material

`DAs/` holds the course deliverables and a reference paper
(`paper_for_tile_defect_class.pdf`) on tile defect classification that informed
the approach. `DAs/` = the Digital Assignment / term-paper submissions for the
course; it is documentation, not code.
