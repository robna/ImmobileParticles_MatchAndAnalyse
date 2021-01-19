import pandas as pd
import numpy as np


def particle_melting(particle_results):
    particle_results.rename(columns={'preIndex': '_preIndex',
                                     'wafer': '_wafer',
                                     'polymer': '_polymer',
                                     'treatment': '_treatment',
                                     'postIndex': '_postIndex'}, inplace=True)
    pa = pd.DataFrame(particle_results.values, columns=particle_results.columns.str.rsplit('_', 1, expand=True))

    pam = pa.melt(id_vars=[
        ('', 'wafer'),
        ('', 'polymer'),
        ('', 'treatment'),
        ('', 'preIndex'),
        ('', 'postIndex')],
                    value_vars=[
                        ('area', 'pre'),
                        ('area', 'post'),
                        ('perimeter', 'pre'),
                        ('perimeter', 'post'),
                        ('intensity', 'pre'),
                        ('intensity', 'post')])

    pam.set_index('variable_1', inplace=True)
    postValuesSeries = pam.loc['post', 'value']
    pam.drop(index='post', inplace=True)
    pam['postValue'] = postValuesSeries.values
    pam.reset_index(drop=True, inplace=True)
    pam.columns = ['wafer', 'polymer', 'treatment', 'preIndex', 'postIndex', 'prop', 'preValue', 'postValue']
    pam['change'] = pd.to_numeric(pam.postValue / pam.preValue - 1, errors='coerce')  # calculate relative change of particle properties for each particle
    # pam.loc[pam.change < - 0.5, 'change'] = np.nan  # exclusion of of obviously wrongly matched particles (experimental)
    # pam.loc[pam.change > 0.5, 'change'] = np.nan  # exclusion of of obviously wrongly matched particles (experimental)
    # pam['matched'] = ~pam.preIndex.isna() & ~pam.postIndex.isna()
    # pam.drop(pam[pam.polymer == 'PVC'].index, inplace=True)

    return pam