import os
import shutil

input_root = "DATASET/Phone/PROCESSED DATA"
output_root = "DATASET/Phone/final"
os.makedirs(output_root, exist_ok=True)

# Helper to check if all 4 rotations are available
def has_all_rotations(base_path, base_name):
    return all(
        os.path.exists(os.path.join(base_path, f"{base_name}_r{angle}.jpg"))
        for angle in [0, 90, 180, 270]
    )

# Process each folder like A, B, etc.
for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue

    # Group all _r0 files to identify image sets
    all_files = sorted(os.listdir(folder_path))
    r0_files = [f for f in all_files if "_r0.jpg" in f]

    pic_count = 1
    for r0_file in r0_files:
        base_name = r0_file.replace("_r0.jpg", "")
        if not has_all_rotations(folder_path, base_name):
            print(f"[SKIP] Missing rotations for {base_name} in {folder}")
            continue

        for angle in [0, 90, 180, 270]:
            src_file = os.path.join(folder_path, f"{base_name}_r{angle}.jpg")
            new_name = f"{folder}_pic{pic_count}_r{angle}.jpg"
            dest_file = os.path.join(output_root, new_name)

            shutil.copy(src_file, dest_file)
            print(f"[OK] {new_name}")

        pic_count += 1
