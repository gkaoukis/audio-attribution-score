import yaml
import soundfile as sf
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_audio_properties(file_path: Path):
    """Reads the WAV header to verify it is 16kHz, mono, PCM_16."""
    errors = []
    try:
        info = sf.info(str(file_path))
        
        if info.samplerate != 16000:
            errors.append(f"Sample rate is {info.samplerate} Hz (Expected: 16000)")
        
        if info.channels != 1:
            errors.append(f"Channels is {info.channels} (Expected: 1/Mono)")
            
        if info.subtype != 'PCM_16':
            errors.append(f"Subtype is '{info.subtype}' (Expected: 'PCM_16')")
            
        # Optional: ensure it's actually a WAV container
        if info.format != 'WAV':
            errors.append(f"Container format is '{info.format}' (Expected: 'WAV')")
            
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

def get_directories_from_config(yaml_path: str):
    """Parses the YAML config and extracts all target audio directories."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dirs_to_check = []
    datasets = config.get('datasets', {})
    
    # 1. FakeMusicCaps
    if 'fakemusiccaps' in datasets:
        dirs_to_check.append(datasets['fakemusiccaps'].get('audio_dir'))
        
    # 2. SMP Dataset
    if 'smp' in datasets:
        dirs_to_check.append(datasets['smp'].get('base_dir'))
        
    # 3. SONICS Dataset
    if 'sonics' in datasets:
        sonics = datasets['sonics']
        if 'sampled' in sonics:
            if 'real' in sonics['sampled']:
                dirs_to_check.append(sonics['sampled']['real'].get('audio_dir'))
            if 'fake' in sonics['sampled']:
                dirs_to_check.append(sonics['sampled']['fake'].get('audio_dir'))
        if 'pairs' in sonics:
            dirs_to_check.append(sonics['pairs'].get('base_dir'))
            
    # Filter out empty entries and convert to Path objects
    return [Path(d) for d in dirs_to_check if d]

def main():
    config_path = "data_config.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found at: {config_path}")
        return

    directories = get_directories_from_config(config_path)
    
    all_wavs = []
    for d in directories:
        if not d.exists():
            logger.warning(f"Directory not found on disk, skipping: {d}")
            continue
            
        # Using rglob matches files in the base dir AND any subfolders (needed for SMP and Pairs)
        wavs = list(d.rglob("*.wav"))
        logger.info(f"Found {len(wavs)} .wav files in {d}")
        all_wavs.extend(wavs)
        
    if not all_wavs:
        logger.error("No .wav files found in the specified directories.")
        return

    # Verify all files
    invalid_files = {}
    logger.info(f"Verifying {len(all_wavs)} files for 16kHz, mono, PCM_16...")
    
    for wav_path in tqdm(all_wavs, desc="Checking audio headers"):
        is_valid, errors = check_audio_properties(wav_path)
        if not is_valid:
            invalid_files[str(wav_path)] = errors
            
    # Print the final report
    print("="*60)
    print(f"Total files checked : {len(all_wavs)}")
    print(f"Valid files         : {len(all_wavs) - len(invalid_files)}")
    print(f"Invalid files       : {len(invalid_files)}")
    print("="*60)
    
    if invalid_files:
        print("\nThe following files do not match the required format:\n")
        for path, errors in invalid_files.items():
            print(f"FILE: {path}")
            for err in errors:
                print(f"  - {err}")
            print()
    else:
        print("\nAll files are perfectly formatted (16kHz, Mono, PCM_16)!\n")

if __name__ == "__main__":
    main()