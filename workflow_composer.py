from pathlib import Path
import time
import numpy as np
import pandas as pd
import concurrent.futures

import runParticleMatching as pm

t_start = time.time()
t_start_formatted = time.strftime('%H:%M:%S', time.localtime(t_start))
print(f'Start time is:   {t_start_formatted}')

keys = pd.read_csv('wafer-polymer-keyfile.csv', index_col='wafer')
keys.dropna(inplace=True)
keys.sort_values(by=['polymer', 'treatment'], inplace=True)  # sort the keys table after polymer and treatment

wafer_results = keys.assign(pre_count=np.nan, post_count=np.nan, matched_count=np.nan,
                            process_time=np.nan)  # prepare a results dataframe
particle_results = pd.DataFrame()

pre_paths = [item for item in Path(  # pathlib.Path.glob creates a generator, which is used to make a list of paths here
    '/run/media/nibor/data_ext/quantDigest_imageData/tif_pre/').glob(  # enter path to pre image directory
    '*.tif')]  # collect all tiff files from path

pre_paths = sorted(pre_paths, key=lambda x: keys.index.get_loc(x.stem.split('_')[0]))  # sort paths like the key table


def process_image_pair(pre_image_path):
    t0 = time.time()

    cwn = pre_image_path.stem.split('_')[0]  # get current wafer name
    cwp = keys.loc[cwn]['polymer']  # get current wafer polymer
    cwt = keys.loc[cwn]['treatment']  # get current wafer treatment

    post_image_path = f'/run/media/nibor/data_ext/quantDigest_imageData/tif_post/{cwn}_{cwt}.tif'

    statsBefore, statsAfter, indexMap, ratios = pm.runPM(pre_image_path, post_image_path)
    indexMap_df = pd.DataFrame(indexMap, index=['ID_post']).transpose()
    indexMap_df.reset_index(inplace=True)
    indexMap_df.rename(columns={'index': 'ID_pre'}, inplace=True)

    if len(statsBefore) > len(statsAfter):
        indexMap_df.columns = indexMap_df.columns[::-1]

    indexMap_df.set_index('ID_pre', inplace=True)

    statsBefore.insert(0, 'file', pre_image_path)
    statsBefore.insert(0, 'state', "pre")
    statsBefore.insert(0, 'polymer', cwp)
    statsBefore.insert(0, 'wafer', cwn)

    statsBefore_wMap = statsBefore.join(indexMap_df, lsuffix='_pre')
    stats_combined = statsBefore_wMap.merge(statsAfter, left_on='ID_post', right_index=True, how='outer')

    statsAfter.insert(0, 'file', post_image_path)
    statsAfter.insert(0, 'state', cwt)
    statsAfter.insert(0, 'polymer', cwp)
    statsAfter.insert(0, 'wafer', cwn)

    tn = round((time.time() - t0) / 60, ndigits=1)
    t0 = time.time()

    return tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexMap, ratios


if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_image_pair, pre_paths)

        for result in results:
            tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexMap, ratios = result

            print(f'Images of {cwn} with {cwp} and {cwt} were completed after {tn} min.')

            wafer_results.at[cwn, 'pre_count'] = len(statsBefore)
            wafer_results.at[cwn, 'post_count'] = len(statsAfter)
            wafer_results.at[cwn, 'matched_count'] = len(ratios)
            wafer_results.at[cwn, 'process_time'] = tn

            particle_results = particle_results.append(stats_combined)

wafer_results.replace({'Napoly':'SPT', '2c':'Acrylate', '4c':'Epoxy'}, inplace=True)
particle_results.replace({'Napoly':'SPT', '2c':'Acrylate', '4c':'Epoxy'}, inplace=True)

wafer_results.to_csv('../wafer_results_{}.csv'.format(pd.datetime.today().strftime('%d-%m-%y_%H-%M')))
particle_results.to_csv('../particle_results_{}.csv'.format(pd.datetime.today().strftime('%d-%m-%y_%H-%M')))

t_final = round((time.time() - t_start) / 3600, ndigits=1)
print(f'All images processed. Total duration was {t_final} h.')
