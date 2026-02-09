import zipfile
from pathlib import Path
from tqdm import tqdm

ZIP_DIR = Path("nse_fo_bhavcopies")
EXTRACT_DIR = Path("nse_fo_bhavcopies_extracted")

EXTRACT_DIR.mkdir(exist_ok=True)

zip_files = list(ZIP_DIR.glob("*.zip"))

print(f"\nFound {len(zip_files)} zip files.\n")

for zip_path in tqdm(zip_files):
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Usually each zip has ONE csv
            for member in z.namelist():
                target_file = EXTRACT_DIR / member

                # Skip if already extracted
                if target_file.exists():
                    continue

                z.extract(member, EXTRACT_DIR)

    except zipfile.BadZipFile:
        print(f"\n⚠️ Corrupt zip skipped: {zip_path.name}")

print("\n✅ Extraction complete.")
