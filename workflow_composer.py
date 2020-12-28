from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import concurrent.futures

import runParticleMatching as pm

# %%
# These keys are required for each process, so we leave it here to have it accessible as global variable in each process..
keys = pd.read_csv('wafer-polymer-keyfile.csv', index_col='wafer')
keys.dropna(inplace=True)
keys.sort_values(by=['polymer', 'treatment'], inplace=True)  # sort the keys table after polymer and treatment

# pre_directory = r'C:\Users\xbrjos\Desktop\New folder\quantDigest_imageData\tif_pre_test'  # Josefs paths
# post_directory = r'C:\Users\xbrjos\Desktop\New folder\quantDigest_imageData\tif_post'  # Josefs paths
pre_directory = r'/run/media/nibor/data_ext/quantDigest_imageData/tif_pre/'  # Robins paths
post_directory = r'/run/media/nibor/data_ext/quantDigest_imageData/tif_post/'  # Robins paths

# %%
def process_image_pair(pre_image_path):
    t0 = time.time()

    cwn = pre_image_path.stem.split('_')[0]  # get current wafer name
    cwp = keys.loc[cwn]['polymer']  # get current wafer polymer
    cwt = keys.loc[cwn]['treatment']  # get current wafer treatment

    post_image_path = os.path.join(post_directory, f'{cwn}_{cwt}.tif')

    statsBefore, statsAfter, indexBefore2After, *_ = pm.runPM(pre_image_path, post_image_path)

    indexMap_df = pd.DataFrame(indexBefore2After, index=['postIndex']).transpose()
    indexMap_df.reset_index(inplace=True)
    indexMap_df.rename(columns={'index': 'preIndex'}, inplace=True)
    indexMap_df.set_index('preIndex', inplace=True)

    statsBefore.insert(0, 'treatment', cwt)
    statsBefore.insert(0, 'polymer', cwp)
    statsBefore.insert(0, 'wafer', cwn)

    statsAfter.insert(0, 'treatment', cwt)
    statsAfter.insert(0, 'polymer', cwp)
    statsAfter.insert(0, 'wafer', cwn)

    statsBefore_wMap = statsBefore.join(indexMap_df)
    stats_combined = statsBefore_wMap.merge(statsAfter,
                                            left_on='postIndex',
                                            right_index=True,
                                            how='outer',
                                            suffixes=('_pre', '_post')
                                            )

    for col in ['wafer', 'polymer', 'treatment']:
        stats_combined[col + '_pre'].fillna(stats_combined[col + '_post'], inplace=True)
        stats_combined.rename(columns={col+'_pre': col}, inplace=True)
        stats_combined.drop(columns=[col+'_post'], inplace=True)

    stats_combined.reset_index(inplace=True)
    stats_combined.rename(columns={'index': 'preIndex'}, inplace=True)

    tn = round((time.time() - t0) / 60, ndigits=1)
    return tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexBefore2After  # , *ratios


# %%
if __name__ == '__main__':
    t_start = time.time()
    t_start_formatted = time.strftime('%H:%M:%S', time.localtime(t_start))
    print(f'Start time is:   {t_start_formatted}')

    # prepare a results dataframe
    wafer_results = keys.assign(pre_count=np.nan, post_count=np.nan, matched_count=np.nan, process_time=np.nan)
    particle_results = pd.DataFrame()
    particle_snips = pd.DataFrame()

    pre_paths = [item for item in
                 Path(  # pathlib.Path.glob creates a generator, which is used to make a list of paths here
                     pre_directory).glob(  # enter path to pre image directory
                     '*.tif')]  # collect all tiff files from path
    pre_paths = sorted(pre_paths,
                       key=lambda x: keys.index.get_loc(x.stem.split('_')[0]))  # sort paths like the key table



    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_image_pair, pre_paths)
        for result in results:
            tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexBefore2After, *ratios = result

            wafer_results.at[cwn, 'pre_count'] = len(statsBefore)
            wafer_results.at[cwn, 'post_count'] = len(statsAfter)
            wafer_results.at[cwn, 'matched_count'] = len(indexBefore2After)
            wafer_results.at[cwn, 'process_time'] = tn

            particle_snip = stats_combined[[
                'wafer', 'polymer', 'treatment',
                'preIndex', 'postIndex',
                'snipW_pre', 'snipH_pre', 'snip_pre',
                'snipW_post', 'snipH_post', 'snip_post'
            ]].copy()
            particle_snips = particle_snips.append(particle_snip)

            stats_combined = stats_combined.drop(['snip_pre', 'snip_post'], axis=1)

            particle_results = particle_results.append(stats_combined)

            print(f'Images of {cwn} with {cwp} and {cwt} were completed after {tn} min.')

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
