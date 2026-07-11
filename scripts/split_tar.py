import os
import sys

def split_file(filepath, chunk_size=48*1024*1024):
    print(f"Splitting {filepath}...")
    with open(filepath, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_name = f"{filepath}.part{chunk_num:03d}"
            with open(part_name, 'wb') as p:
                p.write(chunk)
            chunk_num += 1
    os.remove(filepath)
    print(f"Removed original {filepath}")

dir_path = r"D:\DEV\US-Stock\GraphFactorFactory\data\3day_cano_aff1ff2"
for f in os.listdir(dir_path):
    if f.endswith(".tar.gz"):
        split_file(os.path.join(dir_path, f))
print("Done!")
