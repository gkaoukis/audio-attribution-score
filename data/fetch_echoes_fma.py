import pandas as pd
import os
import urllib.request
import zipfile
import fsspec

# 1. Load your dataset manifest
print("Parsing dataset_manifest.csv...")
df = pd.read_csv('dataset_manifest.csv')
original_audio = df['original_audio'].unique()
print(f"Found {len(original_audio)} unique original tracks to download.")

# 2. Download and extract FMA metadata (~342 MB) to get track IDs
metadata_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
if not os.path.exists("fma_metadata/tracks.csv"):
    if not os.path.exists("fma_metadata.zip"):
        print("Downloading FMA metadata (this may take a minute)...")
        urllib.request.urlretrieve(metadata_url, "fma_metadata.zip")
    
    print("Extracting tracks.csv...")
    with zipfile.ZipFile("fma_metadata.zip", 'r') as zip_ref:
        zip_ref.extract("fma_metadata/tracks.csv", path=".")

# 3. Load the FMA tracks.csv (FMA uses a multi-index header)
print("Mapping FMA track IDs...")
tracks_df = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0, 1])

track_mapping = {}
for track_id, row in tracks_df.iterrows():
    title = str(row[('track', 'title')])
    artist = str(row[('artist', 'name')])
    formatted_name = f"{title} - {artist}"
    track_mapping[formatted_name] = track_id

# 4. Find the FMA track_ids for your specific dataset tracks
target_ids = {}
for name in original_audio:
    if name in track_mapping:
        target_ids[name] = track_mapping[name]
    else:
        print(f"Warning: Could not find exact match for '{name}' in FMA metadata.")

print(f"\nSuccessfully matched {len(target_ids)} out of {len(original_audio)} tracks.")

# 5. Stream ONLY the required MP3 files from the 900GB fma_large.zip
print("\nConnecting to fma_large.zip via HTTP Range Requests...")
os.makedirs("original_fma_tracks", exist_ok=True)

# fsspec's HTTPFileSystem allows zipfile to seek and read the central directory at the end of the remote file,
# allowing us to download only the exact bytes for the specific MP3s we want.
fs = fsspec.filesystem('http')
fma_large_url = 'https://os.unil.cloud.switch.ch/fma/fma_large.zip'

with fs.open(fma_large_url, block_size=1024*1024) as f:
    with zipfile.ZipFile(f) as zf:
        print("Fetching remote zip directory listing...")
        namelist = set(zf.namelist())
        
        # Download each matched file
        for name, tid in target_ids.items():
            # FMA internal structure is based on the padded track ID: e.g., '000/000002.mp3'
            tid_str = f"{tid:06d}"
            folder = tid_str[:3]
            
            # Check for the correct internal zip path
            zip_path_1 = f"{folder}/{tid_str}.mp3"
            zip_path_2 = f"fma_large/{folder}/{tid_str}.mp3"
            
            if zip_path_1 in namelist:
                zip_path = zip_path_1
            elif zip_path_2 in namelist:
                zip_path = zip_path_2
            else:
                print(f"  [Skipped] {tid_str}.mp3 not found inside remote zip.")
                continue
            
            # Create a safe filename for local storage
            safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in " -_,."]).rstrip()
            local_filename = f"echoes/original_fma_tracks/{safe_name}.mp3"
            
            if os.path.exists(local_filename):
                print(f"  [Exists] {local_filename}")
                continue
                
            try:
                print(f"  [Downloading] {name} ({zip_path})...")
                file_data = zf.read(zip_path)
                with open(local_filename, 'wb') as out_file:
                    out_file.write(file_data)
            except Exception as e:
                print(f"  [Failed] to download {name}: {e}")

print("\nDownload complete! Check the 'original_fma_tracks' folder.")