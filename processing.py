import os
import time
import pandas as pd
from pathlib import Path
from typing import List

import runParticleMatching as pm


def getTifFilesFromDirectory(dirPath: str, keys: pd.DataFrame) -> List[Path]:
    # pathlib.Path.glob creates a generator, which is used to make a list of paths here
    paths: List[Path] = [item for item in Path(dirPath).glob('*.tif')]
    paths = sorted(paths, key=lambda x: keys.index.get_loc(x.stem.split('_')[0]))  # sort paths like the key table
    return paths


def process_image_pair(pre_image_path):
    from compareDigestImages import keys, post_directory
    t0 = time.time()

    cwn = pre_image_path.stem.split('_')[0]  # get current wafer name
    cwp = keys.loc[cwn]['polymer']  # get current wafer polymer
    cwt = keys.loc[cwn]['treatment']  # get current wafer treatment

    post_image_path = os.path.join(post_directory, f'{cwn}_{cwt}.tif')

    statsBefore, statsAfter, indexBefore2After, beforeMax, afterMax, *imgOverlays = pm.runParticleMatching(pre_image_path, post_image_path)

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
    return tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexBefore2After,\
           beforeMax, afterMax, imgOverlays if 'imgOverlays' in locals() else None


def process_results(results, wafer_results, particle_results, particle_snips):
    from compareDigestImages import t_start
    for result in results:
        tn, cwn, cwp, cwt, statsBefore, statsAfter, stats_combined, indexBefore2After, beforeMax, afterMax, *imgOverlays = result
        wafer_results.at[cwn, 'pre_count'] = len(statsBefore)
        wafer_results.at[cwn, 'post_count'] = len(statsAfter)
        wafer_results.at[cwn, 'matched_count'] = len(indexBefore2After)
        wafer_results.at[cwn, 'pre_histBGpeak'] = beforeMax[0]
        wafer_results.at[cwn, 'post_histBGpeak'] = afterMax[0]
        wafer_results.at[cwn, 'pre_histFGpeak'] = beforeMax[1]
        wafer_results.at[cwn, 'post_histFGpeak'] = afterMax[1]
        wafer_results.at[cwn, 'process_time'] = tn
        if imgOverlays[0][0] is not None:
            pre_imgOverlay = imgOverlays[0][0]['pre_imgOverlay']
            post_imgOverlay = imgOverlays[0][0]['post_imgOverlay']
            wafer_results.at[cwn, 'pre_image'] = pre_imgOverlay
            wafer_results.at[cwn, 'post_image'] = post_imgOverlay

        particle_snip = stats_combined[[
            'wafer', 'polymer', 'treatment',
            'preIndex', 'postIndex',
            'snipW_pre', 'snipH_pre', 'snip_pre',
            'snipW_post', 'snipH_post', 'snip_post'
        ]].copy()
        particle_snips = particle_snips.append(particle_snip)

        stats_combined = stats_combined.drop(['snip_pre', 'snip_post'], axis=1)

        particle_results = particle_results.append(stats_combined)

        print(f'Images of {cwn} with {cwp} and {cwt} were completed after {tn} min. Total time so far: {round((time.time() - t_start) / 3600, ndigits=1)} h')

    return wafer_results, particle_results, particle_snips
