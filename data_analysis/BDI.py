import pandas as pd
import numpy as np
from scipy import stats
from settings import Config


def optimise(df):
        # df = wafer_results[['wafer', 'polymer', 'treatment',
    #                         #'pre_count',
    #                         # 'post_count', 'matched_count',
    #                         'pre_histBGpeak', 'post_histBGpeak', 'pre_histFGpeak', 'post_histFGpeak',
    #                         # 'pre_image', 'post_image', 'process_time',
    #                         # 'pre_area_matched', 'post_area_matched',
    #                         'particle_loss', 'area_change', 'mode',
    #                         'pre_histDelta', 'post_histDelta', 'histDeltaDiff'
    #                     ]]
    # dfc = df.loc[df['mode'] == 'water_corrected']
    # df = df[df['mode'] != 'water_corrected']
    # df.area_change = abs(df.area_change)

    df_r = pd.DataFrame(columns=['alpha', 'beta',
                                 'r_BDI', 'p_BDI',
                                 'r_IQI', 'p_IQI'])

    p = np.arange(Config.minParameterForBDI, Config.maxParameterForBDI, Config.stepParameterForBDI)
    for alpha in p:
        for beta in p:
            if alpha == 0 and beta ==0:
                pass
            else:
                df = make_BDI(df, alpha, beta)
                r_BDI = stats.pearsonr(df.particle_loss, df.BDI)
                r_IQI = stats.pearsonr(df.particle_loss, df.IQI)
                df_r.loc[len(df_r)] = [alpha, beta,
                                       r_BDI[0], r_BDI[1],
                                       r_IQI[0], r_IQI[1]]
                print(f'Running BDI optimisation with alpha =     {round(alpha, 2)}                        ', end="\r", flush=True)

    # print(df_r.loc[df_r.r_IQI == df_r.r_IQI.max()])
    print(df_r.loc[df_r.r_BDI == df_r.r_BDI.max()])
    bestAlpha, bestBeta = df_r.loc[df_r.r_BDI == df_r.r_BDI.max()].iloc[0, 0:2]

    return bestAlpha, bestBeta


def make_BDI(df, alpha=1, beta=1):

    df['BDI'] = alpha * (df.pre_histBGpeak + df.post_histBGpeak) + beta * abs(
        df.pre_histBGpeak - df.post_histBGpeak)
    df['IQI'] = alpha * (df.post_histDelta + df.pre_histDelta) - beta * df.histDeltaDiff
    # df['IQI'] = df['BDI'].max() - df['BDI']

    return df

