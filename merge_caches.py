import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    cpu_dir = Path("feature_cache_cpu")
    gpu_dir = Path("feature_cache_gpu")
    out_dir = Path("feature_cache_merged")
    out_dir.mkdir(exist_ok=True)

    cpu_files = list(cpu_dir.glob("*.npz"))
    print(f"Merging {len(cpu_files)} feature files into {out_dir}...")

    for cpu_file in tqdm(cpu_files):
        gpu_file = gpu_dir / cpu_file.name
        
        # Load CPU data (classical, ai_detection, fakeprint)
        data = dict(np.load(cpu_file, allow_pickle=True))
        
        # Merge GPU data if it exists (mert, clap)
        if gpu_file.exists():
            gpu_data = dict(np.load(gpu_file, allow_pickle=True))
            data.update(gpu_data)
            
        # Save to the new merged folder
        np.savez_compressed(out_dir / cpu_file.name, **data)

    # Copy and update the metadata file so paths don't break
    meta_in = cpu_dir / "feature_metadata.csv"
    meta_out = out_dir / "feature_metadata.csv"
    
    if meta_in.exists():
        df = pd.read_csv(meta_in)
        # Update the path to point to the new merged directory
        df['cache_path'] = df['cache_path'].apply(lambda x: str(out_dir / Path(x).name))
        df.to_csv(meta_out, index=False)
        print(f"\nMetadata successfully updated and saved to {meta_out}")
    else:
        print("\nWarning: feature_metadata.csv not found in CPU directory.")

    print("Merge complete! You are ready to train.")

if __name__ == "__main__":
    main()