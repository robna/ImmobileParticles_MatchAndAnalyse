"""
The main script for comparing a set of images from before, and after certain particle digest procedures
"""
import time
from datetime import datetime
import pandas as pd
from typing import List
from pathlib import Path
import concurrent.futures

from processing import process_image_pair, process_results, getTifFilesFromDirectory
from runParticleMatching import getPxAreaOfEquivalentSphere
from outputs import prepareDefaultWaferDataFrame

t_start = time.time()

px_res_orig = 0.359  # µm / px in original resolution of microscope (value from image meta data: 0.35934295644272635)
px_res = px_res_orig * 2  # 2nd level of pyramid of source image was used, so double resolution.

keys: pd.DataFrame = pd.read_csv('wafer-polymer-keyfile.csv', index_col='wafer')
keys.dropna(inplace=True)
keys.sort_values(by=['polymer', 'treatment'], inplace=True)  # sort the keys table after polymer and treatment

pre_directory = r'/insert_path_to_images/tif_pre/'  # adjust to where the pre-treatment images are saved
post_directory = r'/insert_path_to_images/tif_post/'  # adjust to where the post-treatment images are saved


class Config:
    """
    Use this struct to specify parameters for image processing and evaluation
    """
    imgScaleFactor: float = 1.0  # (0...1.0)
    minParticleArea: int = getPxAreaOfEquivalentSphere(10, px_res, imgScaleFactor)
    maxParticleArea: int = getPxAreaOfEquivalentSphere(200, px_res, imgScaleFactor)
    hystHighThresh: float = 0.75  # relative to maximum image intensity (0...1.0)
    showPartImages: bool = True  # whether or not to show the found and paired particles in before and after image
    multiprocessing: bool = False


if __name__ == '__main__':
    t_start_formatted = time.strftime('%H:%M:%S', time.localtime(t_start))
    print(f'Start time is:   {t_start_formatted}')

    wafer_results: pd.DataFrame = prepareDefaultWaferDataFrame(keys)
    particle_results: pd.DataFrame = pd.DataFrame()
    particle_snips: pd.DataFrame = pd.DataFrame()  # Snips are single particle images cut out from the wafer photos

    pre_paths: List[Path] = getTifFilesFromDirectory(pre_directory, keys)

    if Config.multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(process_image_pair, pre_paths)
            wafer_results, particle_results, particle_snips = process_results(results, wafer_results, particle_results, particle_snips)

    else:
        results = map(process_image_pair, pre_paths)
        wafer_results, particle_results, particle_snips = process_results(results, wafer_results, particle_results, particle_snips)

    # Some name fixing here, for easier handling afterwards
    wafer_results.replace({'pentane': 'Pentane', 'Napoly': 'SPT', '2c': 'Acrylate', '4c': 'Epoxy'}, inplace=True)
    particle_results.replace({'pentane': 'Pentane', 'Napoly': 'SPT', '2c': 'Acrylate', '4c': 'Epoxy'}, inplace=True)
    particle_snips.replace({'pentane': 'Pentane', 'Napoly': 'SPT', '2c': 'Acrylate', '4c': 'Epoxy'}, inplace=True)

    wafer_results.to_csv('results_csv/wafer_results_{}.csv'.format(datetime.today().strftime('%d-%m-%y_%H-%M')))
    particle_results.to_csv(
        'results_csv/particle_results_{}.csv'.format(datetime.today().strftime('%d-%m-%y_%H-%M')), index=False)
    particle_snips.to_csv(
        'results_csv/particle_snips_{}.csv'.format(datetime.today().strftime('%d-%m-%y_%H-%M')), index=False)

    t_final = round((time.time() - t_start) / 3600, ndigits=1)
    print(f'All images processed. Total duration was {t_final} h.')
