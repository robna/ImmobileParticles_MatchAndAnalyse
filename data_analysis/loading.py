from pathlib import Path
import pandas as pd

def latest_file(path: Path, pattern: str = "*"):
    files = path.glob(pattern)
    return max(files, key=lambda x: x.stat().st_ctime)


def load_csv_results(path):

    particle_path = latest_file(Path(path),'particle_results*.csv')
    print (f'Particle file used is:   {particle_path.name}')
    particle_results = pd.read_csv(particle_path)

    wafer_path = latest_file(Path(path),'wafer_results*.csv')
    print (f'Wafer file used is:   {wafer_path.name}')
    wafer_results = pd.read_csv(wafer_path)

    snips_path = latest_file(Path(path),'particle_snips*.csv')
    print (f'Snips file used is:   {snips_path.name}')
    particle_snips = pd.read_csv(snips_path).dropna()  # dropna: only keep snips of particles that were matched

    return wafer_results, particle_results, particle_snips



