import os
import tarfile
from glob import glob

base_dir = r"D:\DEV\US-Stock\GraphFactorFactory\data\graph_store\canonical"
output_dir = r"D:\DEV\US-Stock\GraphFactorFactory\data\daily_cano_packs"
os.makedirs(output_dir, exist_ok=True)

CHUNK_SIZE = 48 * 1024 * 1024

for folder in sorted(glob(os.path.join(base_dir, "date=2026-01-*"))):
    date_str = os.path.basename(folder).split('=')[1]
    tar_path = os.path.join(output_dir, f"cano_{date_str}.tar.gz")
    
    # Check if this day is already packed and split (check for part001)
    if os.path.exists(f"{tar_path}.part001"):
        print(f"Skipping {date_str}, already packed.")
        continue
    
    print(f"Archiving {date_str}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(folder, arcname=os.path.basename(folder))
    
    print(f"Splitting {date_str} into 48MB chunks...")
    chunk_num = 1
    with open(tar_path, "rb") as f_in:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            part_path = f"{tar_path}.part{chunk_num:03d}"
            with open(part_path, "wb") as f_out:
                f_out.write(chunk)
            chunk_num += 1
            
    # Remove the un-split tar to save space
    os.remove(tar_path)
    print(f"Finished {date_str}.")

print("All daily packs generated successfully.")
