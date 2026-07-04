import glob
import pyarrow.parquet as pq
import concurrent.futures

def check(p):
    try:
        pq.read_table(p, columns=['symbol_id'])
        return None
    except Exception as e:
        return p

def main():
    print("Checking P drive...")
    files = glob.glob(r'P:\US-Stock\NodeFactorFactory\warehouse\month_packs\month=2026-06\**\*.parquet', recursive=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        bad = [p for p in ex.map(check, files) if p is not None]
    print(f'Total files: {len(files)}, Corrupted: {len(bad)}')
    if bad:
        print('First 5 bad:', bad[:5])

if __name__ == '__main__':
    main()
