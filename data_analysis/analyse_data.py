from pathlib import Path
from loading import load_csv_results as load
from wafer_data_wrangling import wafer_wrangling
from particle_data_wrangling import particle_melting
from dashboard import make_dashboard

inPath = Path('../results_csv/')
outPath = Path('../figures/')

wafer_results, particle_results, particle_snips = load(inPath)

molten_particles = particle_melting(particle_results)

waffles, wafer_images, paramsDF = wafer_wrangling(wafer_results, molten_particles)

if __name__ == '__main__':
    make_dashboard(waffles, molten_particles, particle_snips, wafer_images, outPath)
