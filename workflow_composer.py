import pathlib
import pandas as pd
import concurrent.futures

import runParticleMatching as pm

keys = pd.read_csv('wafer-polymer-keyfile.csv', index_col='wafer')

pre_paths = pathlib.Path(
    '/run/media/nibor/data_ext/quantDigest_imageData/tif_pre_test/').rglob(
    '*.tif')  # collect all tiff files from path
pre_paths = sorted([x for x in pre_paths])


def process_image_pair(pre_image_path):
    cwn = pre_image_path.stem.split('_')[0]  # get current wafer name
    post_image_path = pathlib.Path('/run/media/nibor/data_ext/quantDigest_imageData/tif_post/').rglob('{cwn}*.tif')
    cwp = keys.loc[cwn]['polymer']  # get current wafer polymer
    cwt = keys.loc[cwn]['treatment']  # get current wafer treatment

    # print(pre_image_path, '   ', post_image_path, '    ', cwp)

    statsBefore, statsAfter, indexMap = pm.runPM(pre_image_path, post_image_path)

    statsBefore.insert(0, 'file', pre_image_path)
    statsBefore.insert(0, 'state', "pre")
    statsBefore.insert(0, 'polymer', cwp)
    statsBefore.insert(0, 'wafer', cwn)

    statsBefore.insert(0, 'file', post_image_path)
    statsBefore.insert(0, 'state', cwt)
    statsBefore.insert(0, 'polymer', cwp)
    statsBefore.insert(0, 'wafer', cwn)

    return statsBefore, statsAfter, indexMap


with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(process_image_pair, pre_paths)
