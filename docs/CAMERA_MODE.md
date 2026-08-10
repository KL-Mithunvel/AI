# Camera Mode — Interactive Perspective Crop & Rotation Augmentation

File: `DATA PROCESSING/crop.py`

"Camera mode" is the project's interactive tool for turning **raw rig-camera
photos** of tiles into clean, flat, consistently-oriented training images. It is
the human-in-the-loop counterpart to the automatic segmentation in
`RawProcessing.py`: instead of the computer guessing where the tile is, the
operator clicks the four corners and the tool does a precise perspective crop,
then multiplies each crop into four rotations for data augmentation.

It is called "camera mode" because it reads from and writes to the **`DATASET/Camera/`**
tree (the rig camera source), as opposed to the phone-photo path
(`DATASET/Phone/`) that `prep_data.py` handles.

---

## 1. What it does, at a glance

```text
DATASET/Camera/RAWDATA/<class>/photo.jpg
        │
        │  operator clicks & drags the tile's 4 corners
        ▼
   perspective warp to a flat rectangle  (removes camera angle/keystone)
        │
        │  operator confirms the preview
        ▼
   save 4 rotations: _r0, _r90, _r180, _r270
        ▼
DATASET/Camera/PROCESSED DATA/<class>/photo_r{angle}.jpg
```

- **Input:** `DATASET/Camera/RAWDATA` (walked recursively; subfolder structure,
  e.g. per-class folders, is preserved into the output).
- **Output:** `DATASET/Camera/PROCESSED DATA`, mirroring the input's subfolders.
- **Supported image types:** `.jpg .jpeg .png .bmp .tiff .webp`.

> The output directory is **purged and recreated** on every run
> (`purge_output_dir()` → `shutil.rmtree` then `mkdir`). Anything already in
> `DATASET/Camera/PROCESSED DATA` is deleted when you start. Re-run only when you
> intend to redo the whole set.

---

## 2. Running it

```bash
# From the repo root, with DATASET/Camera/RAWDATA populated:
python "DATA PROCESSING/crop.py"
```

Requires OpenCV with GUI support (`cv2.imshow` windows) — it is an interactive
desktop tool, not a headless/batch script. It raises `FileNotFoundError` if
`DATASET/Camera/RAWDATA` does not exist.

The two folder paths are hard-coded near the top of the file:

```python
FOLDER1 = Path("DATASET/Camera/RAWDATA")          # input
FOLDER2 = Path("DATASET/Camera/PROCESSED DATA")   # output (purged each run)
```

Change these if your capture source lives elsewhere (e.g. point `FOLDER1` at
`DATASET/Phone/RAWDATA` to run the same tool over phone photos).

---

## 3. The operator workflow (per image)

For every image found under `RAWDATA`, the tool opens a corner-selection window:

### 3.1 Selecting the 4 corners — `get_user_points(img)`
- The image is **scaled to fit the screen** (assumes a 1920×1080 display, leaves
  a margin) so large photos are fully visible; all click coordinates are mapped
  back to the original full-resolution image, so the crop is done at full
  quality regardless of the on-screen scale.
- **Click** to place up to 4 red corner dots.
- **Click-and-drag** an existing dot (within a 20 px grab radius) to nudge it —
  useful for fine-tuning corner placement without starting over.
- **Point reuse:** the last-used 4 points are remembered across images
  (`last_used_points`) and pre-loaded for the next photo, as long as they still
  fall inside it. When the rig and tile stay roughly fixed between shots, you
  barely have to re-click — just tweak.
- **Keys:**
  - `Enter` — accept the 4 corners and go to preview (only enabled once exactly
    4 points exist).
  - `Esc` — skip this image entirely.

### 3.2 Confirming the crop — `confirm_and_save(...)`
After `Enter`, the tool warps the selection and shows a **Cropped Preview**
window:

- `s` — **save**: writes the 4 rotations and moves to the next image.
- `b` — **back**: discard and re-select corners for this same image.
- `Esc` — **skip after crop**: don't save, move on.

The loop (`process_all_images`) keeps offering re-selection until you either save
or skip, so a bad crop is always recoverable without re-running the tool.

---

## 4. The image processing

### 4.1 Perspective crop — `perspective_crop(img, points)`
Takes the 4 clicked source points and warps the quadrilateral they enclose into a
**flat, axis-aligned rectangle**:

1. Output width  = max of the top and bottom edge lengths.
2. Output height = max of the left and right edge lengths.
   (Using the max of each opposing pair preserves the largest real extent and
   avoids squashing when the four sides differ due to camera angle.)
3. `cv2.getPerspectiveTransform(src, dst)` builds the homography;
   `cv2.warpPerspective` applies it.

The corner order is assumed to be **top-left → top-right → bottom-right →
bottom-left** (the destination rectangle is built in that order). Click the
corners consistently in that order, or the output orientation will differ.

This removes **keystone/perspective distortion** — the tile comes out as if
photographed straight-on, which is what a downstream classifier wants.

### 4.2 Rotation augmentation — `rotate_image(img, angle)`
On save, the cropped tile is written at **four orientations** — `0°, 90°, 180°,
270°` — using `cv2.rotate`. Each is saved as `"{stem}_r{angle}{suffix}"`, e.g.
`20250912_083243_r0.jpg`, `_r90.jpg`, `_r180.jpg`, `_r270.jpg`.

This is deliberate **data augmentation**: it 4× the dataset for free and makes the
downstream model **rotation-invariant** — a tile's grade shouldn't depend on
which way up it sat on the belt. The naming convention (`_r{angle}`) is what the
next stage (`prep_data.py`) keys off of to group and flatten the set for
Roboflow.

---

## 5. How camera mode fits the pipeline

```text
crop.py (camera mode)                prep_data.py                 CNN1.ipynb
────────────────────                 ────────────                 ──────────
Camera/RAWDATA                       Phone/PROCESSED DATA          PROCESSED_DATA (Drive)
   │  manual 4-corner crop              │  keep only complete         │  CLAHE + high-pass
   │  + 4 rotations                     │  4-rotation sets            │  → grayscale 256²
   ▼                                    │  rename → {cls}_pic{n}_r{a}  ▼
Camera/PROCESSED DATA  ───────────►     ▼                          5-block CNN
                                     final/  (flat, Roboflow-ready) → 4-class grade
```

`crop.py` and `prep_data.py` are written against two different source trees
(`Camera/` vs `Phone/`) but produce the same `_r{angle}` rotation convention, so
either capture source can feed the same training flow. Camera mode is the
higher-effort, higher-quality path (true perspective correction from
hand-placed corners); the phone path assumes crops already exist and only
renames/flattens them.

---

## 6. Practical notes & limitations

- **Interactive only.** Needs a display and mouse; it cannot run on a headless
  server or in CI. Every image requires an operator decision.
- **Destructive output.** The output folder is wiped at the start of each run
  (§1). Back up `PROCESSED DATA` before re-running if you added anything there
  manually.
- **Screen-size assumption.** The fit-to-screen scaling assumes 1920×1080. On a
  different resolution the display may be larger/smaller than intended, though
  the saved crop is unaffected (coordinates are always mapped back to full res).
- **Corner-order dependence.** Consistency of click order (TL→TR→BR→BL)
  determines output orientation; the tool does not auto-sort the four points.
- **No checkerboard awareness.** Unlike `RawProcessing.py`, camera mode doesn't
  know about the calibration checkerboard — the operator simply crops to the tile
  and leaves the checkerboard out by where they click.
- **Reusability:** this is the most directly reusable piece of the old project for
  building a clean tile dataset in the newer `Tile_Sorting` work — the perspective
  correction + rotation augmentation + resumable per-image workflow are all
  generically useful. See [`METHODS.md`](METHODS.md) §7.
