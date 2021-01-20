import pandas as pd
import glm
from settings import Config

# some abbreviations
pl = 'particle_loss'
ac = 'area_change'
gC = '_glmCorrected'
wC = '_waterCorrected'
w = 'water'

def get_areas_from_particles(wafer_results, molten_particles):
    molten_particles_matched_only = molten_particles.dropna()  # take only matched particles: used for calculating aggregated areas
    particle_wafer_groups = molten_particles_matched_only.loc[molten_particles_matched_only.prop == 'area'].groupby('wafer')  # grouping by wafer: only particles that matched between pre and post are used here
    for w, waf in particle_wafer_groups:  # cycle through polymer groups
        pre_area_matched_mean = waf.preValue.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'pre_area_matched'] = round(pre_area_matched_mean)
        post_area_matched_mean = waf.postValue.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'post_area_matched'] = round(post_area_matched_mean)

    wafer_results[pl] = abs(wafer_results.matched_count / wafer_results.pre_count - 1)  # Get normalised particle losses for each wafer: counts of particles in pre state divided by counts in post state. Subtract 1 to normalise. Take absolute value to treat particle loss and particle addition as the same.
    wafer_results[ac] = wafer_results.post_area_matched / wafer_results.pre_area_matched - 1  # same for area...
    return wafer_results


def subtract_water(wafer_results):
    wafer_polymer_groups = wafer_results.groupby('polymer')  # group by polymers
    wafer_results_wrangled = pd.DataFrame()  # create empty df to fill in results from loop

    for _, pol in wafer_polymer_groups:  # cycle through polymer groups
        pol[pl + wC] = pol.loc[pol.treatment != w, pl] - pol.loc[pol.treatment == w, pl].iloc[0]  # subtract particle loss of water from particle losses of all other treatments to get the percentage point error (meaning loss OR addition) of particle numbers caused by each treatment
        pol[ac + wC] = pol.loc[pol.treatment != w, ac] - pol.loc[pol.treatment == w, ac].iloc[0]  # same for area
        pol[pl + gC + wC] = pol.loc[pol.treatment != w, pl + gC] - pol.loc[pol.treatment == w, pl + gC].iloc[0]  # subtract particle loss of water from particle losses of all other treatments to get the percentage point error (meaning loss OR addition) of particle numbers caused by each treatment
        pol[ac + gC + wC] = pol.loc[pol.treatment != w, ac + gC] - pol.loc[pol.treatment == w, ac + gC].iloc[0]  # same for area

        pol.loc[(pol[pl + wC] < 0), pl + wC] = 0  # set all ratios that were smaller than water ratio (and thus got now negative due to water ratio correction) to 0.
        # pol.loc[(pol[ac + wC] > 0), ac + wC] = 0  # same for area
        pol.loc[(pol[pl + gC + wC] < 0), pl + gC + wC] = 0  # set all ratios that were smaller than water ratio (and thus got now negative due to water ratio correction) to 0.
        # pol.loc[(pol[ac + gC + wC] > 0), ac + gC + wC] = 0  # same for area

        wafer_results_wrangled = wafer_results_wrangled.append(pol)  # save results to df
    return wafer_results_wrangled

def semiMelting(wafer_results_wrangled):
    waterCorr_wafer_results = wafer_results_wrangled.drop([pl, ac, pl + gC, ac + gC], axis=1).copy()
    waterCorr_wafer_results.rename(columns={
        pl + wC: pl,
        ac + wC: ac,
        pl + gC + wC: pl + gC,
        ac + gC + wC: ac + gC}, inplace=True)
    waterCorr_wafer_results['mode'] = 'water_corrected'

    nonCorr_wafer_results = wafer_results_wrangled.drop([pl + wC, ac + wC, pl + gC + wC, ac + gC + wC], axis=1).copy()
    nonCorr_wafer_results['mode'] = 'non_corrected'
    semiMelted_wafers = pd.concat([waterCorr_wafer_results, nonCorr_wafer_results])
    return semiMelted_wafers


def meltHelper(df, value_name):
    dfMelt = df.melt(id_vars=['wafer', 'polymer', 'treatment', 'pre_count', 'post_count', 'matched_count',
                              'pre_area_matched', 'post_area_matched', 'BDI', 'IQI',
                              #'pre_image', 'post_image'
                             ],
                    value_vars=['non_corrected', 'glm_corrected', 'water_corrected', 'glm_and_water_corrected'],
                    var_name='mode', value_name=value_name)
    return dfMelt


def nameHelper(df, att):
    dfRenamed = df.rename(columns={att: 'non_corrected',
                                   att + '_glmCorrected': 'glm_corrected',
                                   att + '_waterCorrected': 'water_corrected',
                                   att + '_glmCorrected_waterCorrected': 'glm_and_water_corrected'})
    return dfRenamed


def doubleMelting(wafer_results_wrangled):
    countNamed = nameHelper(wafer_results_wrangled, att='particle_loss')
    countMelt = meltHelper(countNamed, value_name='particle_loss')
    areaNamed = nameHelper(wafer_results_wrangled, att='area_change')
    areaMelt = meltHelper(areaNamed, value_name='area_change')
    areaMelt.drop(columns=['wafer', 'polymer', 'treatment', 'pre_count', 'post_count', 'matched_count',
                           'pre_area_matched', 'post_area_matched', 'BDI', 'IQI',
                           #'pre_image', 'post_image',
                           'mode'], inplace=True)
    doubleMelt = pd.concat([countMelt, areaMelt], axis=1)
    return doubleMelt


def wafer_wrangling(wafer_results, molten_particles):
    wafer_results.dropna(subset=['pre_count'], inplace=True)  # drop any lines were there were no results (i.e. no particles found in pre image)

    # TODO: change dropping procedures to get info what to drop from Config from analyse_data.py
    # wafer_results.drop(
    #     wafer_results.loc[(wafer_results.treatment == 'KOH') & (wafer_results.polymer.isin(['PET', 'PP']))].index,
    #     inplace=True)  # exclude failed KOH wafers
    # wafer_results.drop(wafer_results.loc[wafer_results.polymer.isin(['PVC','PMMA','PS', 'ABS'])].index, inplace=True)  # exclude whole polymers that do not work well with DIC imaging
    # wafer_results.drop(wafer_results[wafer_results.treatment == 'HCl'].index, inplace=True)  # drop calibration HCl treatment (if not wanted in results)

    wafer_images = pd.DataFrame()
    wafer_images[['wafer', 'polymer', 'treatment']] = wafer_results[['wafer', 'polymer', 'treatment']]
    wafer_images['pre_image'] = wafer_results.pop('pre_image')  # separate images into own df
    wafer_images['post_image'] = wafer_results.pop('post_image')

    wafer_results = get_areas_from_particles(wafer_results, molten_particles)  # aggregates particle data to get mean particle areas in wafer df

    if not {'pre_histDelta', 'post_histDelta', 'histDeltaDiff'}.issubset(wafer_results.columns):
        wafer_results['pre_histDelta'] = wafer_results.pre_histFGpeak - wafer_results.pre_histBGpeak
        wafer_results['post_histDelta'] = wafer_results.post_histFGpeak - wafer_results.post_histBGpeak
        wafer_results['histDeltaDiff'] = abs(wafer_results.post_histDelta - wafer_results.pre_histDelta)
        wafer_results['histBGpeakSum'] = wafer_results.pre_histBGpeak + wafer_results.post_histBGpeak
        wafer_results['histBGpeakDist'] = abs(wafer_results.pre_histBGpeak - wafer_results.post_histBGpeak)

    wafer_results, paramsDF = glm.count_loss_glm(wafer_results)  # this fits a glm to particle_loss (binomial data) with pre_count (n) and BDI as predictors. Standard residuals are taken as new (actual treatment-dependent) loss values.
    wafer_results, paramsDF = glm.area_change_glm(wafer_results, molten_particles)

    if Config.manualBDI:
        import BDI
        alpha, beta = BDI.optimise(wafer_results)  # runs a linear regression between differently calculated BDIs and particle_loss, it give back the alpha and beta paramters for the BDI calculation, that produced the highest correlation r value.
        wafer_results = BDI.make_BDI(wafer_results, alpha, beta)  # takes the optimised alpha, beta values and calculates the BDI for all wafers

    wafer_results = subtract_water(wafer_results)  # offset loss and change values by the respective negative control (water treatment)





    if Config.semiMelted:
        wafer_results_wrangled = semiMelting(wafer_results)
    else:
        wafer_results_wrangled = doubleMelting(wafer_results)
    return wafer_results_wrangled, wafer_images, paramsDF
