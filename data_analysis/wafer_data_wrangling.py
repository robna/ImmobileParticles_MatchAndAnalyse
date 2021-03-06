import pandas as pd
import numpy as np
import scipy.stats as stats
import glm
import settings

# some abbreviations
pl = 'particle_loss'
ac = 'area_change'
gC = '_glm_corrected'
wC = '_water_corrected'
w = 'water'
prc = 'pre_count'
poc = 'post_count'
mc = 'matched_count'
pram = 'pre_area_matched'
poam = 'post_area_matched'
# ma = 'matched_area'


def get_areas_from_particles(wafer_results, molten_particles):
    molten_particles_matched_only = molten_particles.dropna()  # take only matched particles: used for calculating aggregated areas
    particle_wafer_groups = molten_particles_matched_only.loc[molten_particles_matched_only.prop == 'area'].groupby('wafer')  # grouping by wafer: only particles that matched between pre and post are used here
    for w, waf in particle_wafer_groups:  # cycle through polymer groups
        pre_area_matched_mean = waf.preValue.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'pre_area_matched'] = pre_area_matched_mean
        post_area_matched_mean = waf.postValue.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'post_area_matched'] = post_area_matched_mean

    wafer_results[pl] = abs(wafer_results.matched_count / wafer_results.pre_count - 1)  # Get normalised particle losses for each wafer: counts of particles in pre state divided by counts in post state. Subtract 1 to normalise. Take absolute value to treat particle loss and particle addition as the same.
    wafer_results[ac] = wafer_results.post_area_matched / wafer_results.pre_area_matched - 1  # same for area...
    return wafer_results


def subtract_water(wafer_results):
    wafer_polymer_groups = wafer_results.groupby('polymer')  # group by polymers
    wafer_results_wrangled = pd.DataFrame()  # create empty df to fill in results from loop

    for _, pol in wafer_polymer_groups:  # cycle through polymer groups
        pol[pl + wC] = pol.loc[pol.treatment != w, pl] - pol.loc[pol.treatment == w, pl].iloc[0]  # subtract particle loss of water from particle losses of all other treatments to get the percentage point error (meaning loss OR addition) of particle numbers caused by each treatment
        pol[ac + wC] = pol.loc[pol.treatment != w, ac] - pol.loc[pol.treatment == w, ac].iloc[0]  # same for area
        pol[pl + gC + '_and' + wC] = pol.loc[pol.treatment != w, pl + gC] - pol.loc[pol.treatment == w, pl + gC].iloc[0]  # subtract particle loss of water from particle losses of all other treatments to get the percentage point error (meaning loss OR addition) of particle numbers caused by each treatment
        pol[ac + gC + '_and' + wC] = pol.loc[pol.treatment != w, ac + gC] - pol.loc[pol.treatment == w, ac + gC].iloc[0]  # same for area

        pol.loc[(pol[pl + wC] < 0), pl + wC] = 0  # set all ratios that were smaller than water ratio (and thus got now negative due to water ratio correction) to 0.
        # pol.loc[(pol[ac + wC] > 0), ac + wC] = 0  # same for area
        pol.loc[(pol[pl + gC + '_and' + wC] < 0), pl + gC + '_and' + wC] = 0  # set all ratios that were smaller than water ratio (and thus got now negative due to water ratio correction) to 0.
        # pol.loc[(pol[ac + gC + '_and' + wC] > 0), ac + gC + '_and' + wC] = 0  # same for area


        mcw = pol.loc[pol.treatment == w, mc]  # matched_count of the water treatment of the current polymer
        mcgCw = pol.loc[pol.treatment == w, mc + gC]  # matched_count of the glm corrected water treatment of the current polymer
        prcw = pol.loc[pol.treatment == w, prc]  # pre_count of the water treatment of the current polymer
        mcts = pol.loc[pol.treatment != w, mc]  # matched_count of all non-water treatments of the current polymer
        mcgCts = pol.loc[pol.treatment != w, mc + gC]
        prcts = pol.loc[pol.treatment != w, prc]
        pol[mc + wC] = round(mcts + prcts - prcts.apply(lambda x: x * mcw / prcw).iloc[:, 0])
        pol[mc + gC + '_and' + wC] = round(mcgCts + prcts - prcts.apply(lambda x: x * mcgCw / prcw).iloc[:, 0])

        pramw = pol.loc[pol.treatment == w, pram]
        poamw = pol.loc[pol.treatment == w, poam]
        poamgCw = pol.loc[pol.treatment == w, poam + gC]
        pramts = pol.loc[pol.treatment != w, pram]
        poamts = pol.loc[pol.treatment != w, poam]
        poamgCts = pol.loc[pol.treatment != w, poam + gC]
        pol[poam + wC] = round(poamts + pramts - pramts.apply(lambda x: x * poamw / pramw).iloc[:, 0])
        pol[poam + gC + '_and' + wC] = round(poamgCts + pramts - pramts.apply(lambda x: x * poamgCw / pramw).iloc[:, 0])

        wafer_results_wrangled = wafer_results_wrangled.append(pol)  # save results to df
    return wafer_results_wrangled


def countAndArea_glmCorrecting(wr):  # wr = wafer_results DF
    wr[mc + gC] = round(wr[prc] * (1 - wr[pl + gC]))
    wr[poam + gC] = round(wr[pram] * (wr[ac + gC] + 1))
    return wr


def semiMelting(wafer_results_wrangled):
    waterCorr_wafer_results = wafer_results_wrangled.drop([pl, ac, pl + gC, ac + gC], axis=1).copy()
    waterCorr_wafer_results.rename(columns={
        pl + wC: pl,
        ac + wC: ac,
        pl + gC + '_and' + wC: pl + gC,
        ac + gC + '_and' + wC: ac + gC}, inplace=True)
    waterCorr_wafer_results['mode'] = 'water_corrected'

    nonCorr_wafer_results = wafer_results_wrangled.drop([pl + wC, ac + wC, pl + gC + '_and' + wC, ac + gC + '_and' + wC], axis=1).copy()
    nonCorr_wafer_results['mode'] = 'non_corrected'
    semiMelted_wafers = pd.concat([waterCorr_wafer_results, nonCorr_wafer_results])
    return semiMelted_wafers


def meltHelper(df, value_name):
    dfMelt = df.melt(id_vars=['wafer', 'polymer', 'treatment', 'pre_count', 'post_count', 'pre_area_matched', 'BDI',
                              'upperPredIntv_confounder_caused_particle_loss',
                              'lowerPredIntv_confounder_caused_particle_loss',
                              'upperPredIntv_confounder_caused_area_change',
                              'lowerPredIntv_confounder_caused_area_change'
                              ],
                    value_vars=['non_corrected', 'glm_corrected', 'water_corrected', 'glm_and_water_corrected'],
                    var_name='mode', value_name=value_name)
    dfMelt.loc[dfMelt['mode'].isin(['non_corrected', 'water_corrected']),
               ['upperPredIntv_confounder_caused_particle_loss',
                'upperPredIntv_confounder_caused_area_change']] = 1000000  # set some high value to not sho pi comparison on uncorrected data
    dfMelt.loc[dfMelt['mode'].isin(['non_corrected', 'water_corrected']),
               ['lowerPredIntv_confounder_caused_particle_loss',
                'lowerPredIntv_confounder_caused_area_change']] = -1000000
    return dfMelt


def nameHelper(df, att):
    dfRenamed = df.rename(columns={att: 'non_corrected',
                                   att + '_glm_corrected': 'glm_corrected',
                                   att + '_water_corrected': 'water_corrected',
                                   att + '_glm_corrected_and_water_corrected': 'glm_and_water_corrected'})
    return dfRenamed


def doubleMelting(wr):
    col_nam = ['particle_loss', 'area_change', 'matched_count', 'post_area_matched', 'count_signiFi']
    doubleMelt = pd.DataFrame()
    for meltee in col_nam:
        named = nameHelper(wr, att=meltee)
        nowMelt = meltHelper(named, value_name=meltee)
        if meltee != 'particle_loss':
            nowMelt.drop(columns=['wafer', 'polymer', 'treatment', 'pre_count', 'post_count', 'pre_area_matched', 'BDI',
                                  'upperPredIntv_confounder_caused_particle_loss',
                                  'lowerPredIntv_confounder_caused_particle_loss',
                                  'upperPredIntv_confounder_caused_area_change',
                                  'lowerPredIntv_confounder_caused_area_change',
                                  'mode'], inplace=True)
        doubleMelt = pd.concat([doubleMelt, nowMelt], axis=1)
    return doubleMelt


def dropping(wafer_results):
    # drop any lines were there were no results (i.e. no particles found in pre image)
    wafer_results.dropna(subset=['pre_count'], inplace=True)

    for d in settings.drops['wafers']:
        wafer_results.drop(wafer_results.loc[(wafer_results.wafer == d)].index, inplace=True)
    for d in settings.drops['combis']:
        wafer_results.drop(wafer_results.loc[(wafer_results.treatment.isin(d)) & \
                                             (wafer_results.polymer.isin(d))].index, inplace=True)
    for d in settings.drops['polymers']:
        wafer_results.drop(wafer_results.loc[(wafer_results.polymer == d)].index, inplace=True)
    for d in settings.drops['treatments']:
        wafer_results.drop(wafer_results.loc[(wafer_results.treatment == d)].index, inplace=True)
        return wafer_results


def predictor_testing(wafer_results, molten_particles):
    # settings.Config.glmPredictorTesting = False
    for f0 in np.arange(0, len(settings.Ndict)):
        settings.Config.glmNpredictor = settings.Ndict[f0]
        for f1 in np.arange(0, len(settings.BDIdict)):
            settings.Config.glmImgQualiPredictors[0] = settings.BDIdict[f1]
            for f2 in np.arange(0, len(settings.BDIdict)):
                settings.Config.glmImgQualiPredictors[1] = settings.BDIdict[f2]

                wafer_results = glm.count_loss_glm(wafer_results)  # this fits a glm to particle_loss (binomial data) with pre_count (n) and BDI as predictors. Standard residuals are taken as new (actual treatment-dependent) loss values.
                wafer_results = glm.area_change_glm(wafer_results, molten_particles)
    return wafer_results


def fisherStats(wr):
    polygroups = wr.groupby('polymer')
    for cm in ['', gC, wC, gC + '_and' + wC]:
        for p, pol in polygroups:
            for tr in pol.treatment.unique():
                prc_tr = pol.loc[pol.treatment == tr, prc].iloc[0]  # pre_count
                success_tr = pol.loc[pol.treatment == tr, mc + cm].iloc[0]  # success = matched_count
                if np.isnan(success_tr) or success_tr > prc_tr:
                    success_tr = prc_tr
                failure_tr = prc_tr - success_tr  # failure = pre_count - success
                prc_w = pol.loc[pol.treatment == w, prc].iloc[0]
                success_w = pol.loc[pol.treatment == w, mc + cm].iloc[0]
                if np.isnan(success_w) or success_w > prc_w:
                    success_w = prc_w
                failure_w = prc_w - success_w
                _, pvalue = stats.fisher_exact(
                    [[success_tr, failure_tr], [success_w, failure_w]],
                    alternative='less')
                wr.loc[(wr.polymer == p) & (wr.treatment == tr), 'count_signiFi' + cm] = pvalue
    return(wr)


def wafer_wrangling(wafer_results, molten_particles):
    wafer_results = dropping(wafer_results)

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
        wafer_results['histDeltaSum'] = wafer_results.pre_histDelta + wafer_results.post_histDelta

    if settings.Config.glmPredictorTesting:
        wafer_results = predictor_testing(wafer_results, molten_particles)
    else:
        wafer_results = glm.count_loss_glm(wafer_results)  # this fits a glm to particle_loss (binomial data) with pre_count (n) and BDI as predictors. Standard residuals are taken as new (actual treatment-dependent) loss values.
        wafer_results = glm.area_change_glm(wafer_results, molten_particles)

    if settings.Config.manualBDI:
        import BDI
        alpha, beta = BDI.optimise(wafer_results)  # runs a linear regression between differently calculated BDIs and particle_loss, it give back the alpha and beta paramters for the BDI calculation, that produced the highest correlation r value.
        wafer_results = BDI.make_BDI(wafer_results, alpha, beta)  # takes the optimised alpha, beta values and calculates the BDI for all wafers
    wafer_results['BDI'] = wafer_results.BDI / wafer_results.BDI.max() * 100  # make BDI go from 0 to 100
    # wafer_results['BDI'] = np.sort(np.asarray(wafer_results.BDI).ravel()).searchsorted(np.asarray(wafer_results.BDI))  # equalise BDI in range

    wafer_results = countAndArea_glmCorrecting(wafer_results)
    wafer_results = subtract_water(wafer_results)  # offset loss and change values by the respective negative control (water treatment)

    wafer_results = fisherStats(wafer_results)

    if settings.Config.semiMelted:
        wafer_results_wrangled = semiMelting(wafer_results)
    else:
        wafer_results_wrangled = doubleMelting(wafer_results)
    return wafer_results_wrangled, wafer_images