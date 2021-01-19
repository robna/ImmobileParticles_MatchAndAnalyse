import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import settings


def prepare_df4glm(wafer_results):
    # df4glm = semiMelted wafers.loc[(semiMelted wafers['mode'] != 'water_corrected'), ['wafer', 'polymer', 'treatment', 'matched_count', 'pre_count', 'BDI']]  # old way, when df4glm was prepared from semiMelted wafers
    df4glm = wafer_results.copy()
    df4glm['failure'] = df4glm.pre_count - df4glm.matched_count
    df4glm['success'] = df4glm['matched_count']
    df4glm = df4glm.loc[df4glm.treatment != 'HCl']
    df4glm = df4glm.loc[df4glm.treatment == 'water']
    return df4glm


def make_glm_arrays(df4glm):
    endog = np.asarray(df4glm[['success', 'failure']])
    exog = np.asarray(df4glm[[settings.Config.glmNpredictor,
                              settings.Config.glmImgQualiPredictors[0],
                              settings.Config.glmImgQualiPredictors[1]
                              ]])
    exog = sm.add_constant(exog, prepend=False)
    return endog, exog


def make_glm_dataframe(wafer_results):
    df4glm = prepare_df4glm(wafer_results)
    endog = df4glm['success'] / (df4glm['success'] + df4glm['failure'])
    df4glm['SUCCESS'] = endog
    return df4glm


def extrap(xp, x, y, ord=3):
    """
    Interpolation and extrapolation of points using a fitted polynomial curve.
    xp = x values at which to get new y values for
    x = x values at which y values are known
    y = known y values
    ord = polynomial order, default is 3
    """

    z = np.polyfit(x, y, ord)
    p = np.poly1d(z)
    yp = p(xp)
    return yp


def glm_residuals(fitted_model, endog, yAll, yhatAll):
    if endog.ndim > 1:
        y = 1 - (endog[:, 0] / endog.sum(1))  # loss rate = 1 - matched / pre
    else:
        y = endog
    yhat = 1 - fitted_model.mu  # fitted loss values for water wafers  TODO: is this correct to take 1 - mu?
    chichi = stats.chisquare(f_obs=yhat, f_exp=y, ddof=0)  # chi² seems not correct...
    # GLM internal Pearson Chi² (for 3 DOF, i.e. GLM with 3 predictors) needs to be above [7.815, 11.345, 16.266] for p [0.05, 0.01, 0.001]

    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()  # make linear regression between original and fitted values
    _, iv_l, iv_u = wls_prediction_std(line_fit)  # calculate predict intervals from linear regression
    iv_lAll = extrap(yhatAll, yhat, iv_l, ord=3)  # get lower pred interval for all point by interpolation / extrapolation
    iv_uAll = extrap(yhatAll, yhat, iv_u, ord=3)  # get upper pred interval for all point by interpolation / extrapolation

    if settings.Config.glmPlots:
        from matplotlib import pyplot as plt
        from textwrap import wrap
        from statsmodels.graphics.api import abline_plot
        fig, ax = plt.subplots(2, 2, figsize=[20, 20])
        # ax.fill_between(yhat, iv_u, iv_l, color="#b9cfe7", edgecolor="")  # attempt to plot predict int as area
        ivs_sorted = np.column_stack([yhatAll.T, iv_uAll.T, iv_lAll.T])  # combine fitted losses with predict ints for sorting
        ivs_sorteds = np.column_stack([yhat.T, iv_u.T, iv_l.T])  # also show original (non-extrapolated) upper pred interval for comparison
        ivs_sorted = ivs_sorted[ivs_sorted[:,0].argsort()]  # sort predict int values by fitted losses to avoid distorted line plot
        ivs_sorteds = ivs_sorteds[ivs_sorteds[:, 0].argsort()]
        ax[0, 0].plot(ivs_sorted[:,0], ivs_sorted[:,1], color='lightgray', linestyle='dashed')  # plot upper predict int
        ax[0, 0].plot(ivs_sorteds[:, 0], ivs_sorteds[:, 1], color='red')
        ax[0, 0].plot(ivs_sorted[:,0], ivs_sorted[:,2], color='lightgray', linestyle='dashed')  # plot lower predict int
        abline_plot(model_results=line_fit, ax=ax[0, 0])  # plot linear regression line
        ax[0, 0].scatter(yhatAll, yAll, s=50)  # plot all original loss values against predicted loss values
        ax[0, 0].scatter(yhat, y)  # plot water point in different color on top
        ax[0, 0].set_xlabel('fitted losses (losses attributed to n and image quality)')
        ax[0, 0].set_ylabel('observed losses')
        ax[0, 0].set_title('Linearized model')

        ax[0, 1].scatter(np.arange(1, len(yAll) + 1, 1), (yAll - yhatAll), s=50)
        # ax[0, 1].scatter(np.arange(1, len(yAll) + 1, len(y)), (y - yhat))
        ax[0, 1].bar(np.arange(1, len(yAll) + 1, 1), (yAll - yhatAll), color='C0')
        ax[0, 1].hlines(0, 0, len(yAll))
        # ax[0, 1].set_ylim(0, 1)
        ax[0, 1].set_xlabel('id')
        ax[0, 1].set_ylabel('standard residuals (corrected losses)')
        ax[0, 1].set_title('Relative losses caused by treatment after GLM correction')

        ax[1, 0].scatter(np.arange(1, len(yAll) + 1, 1), yAll)
        ax[1, 0].scatter(np.arange(1, len(yAll) + 1, 1), yhatAll)
        ax[1, 0].set_xlabel('id')
        ax[1, 0].set_ylabel('observed and fitted losses')
        ax[1, 0].set_title("\n".join(wrap('Original losses (blue) and losses attributed to n and image quality (orange)', 80)))

        ax[1, 1].bar(np.arange(1, len(yAll) + 1, 1), yAll - yhatAll)  #, width=0.2)
        ax[1, 1].bar(np.arange(1, len(yAll) + 1, 1), yhatAll, bottom=yAll - yhatAll)  #, width=0.2)
        ax[1, 1].scatter(np.arange(1, len(yAll) + 1, 1), np.maximum(yAll - yhatAll, 0), color='black', zorder=4)
        ax[1, 1].set_ylim(0, 1)
        ax[1, 1].set_xlabel('id')
        ax[1, 1].set_ylabel('relative losses')
        ax[1, 1].set_title("\n".join(wrap('Original losses (tops) and losses attributed to n and image quality (upper bar) and losses attributed to treatment effects (bottom bar)', 80)))
        fig.show()
        plt.show(block=True)

    if settings.Config.useGLMpredIv:
        glm_corrected_particle_losses = np.round((np.maximum(yAll - iv_uAll, 0)), 2)
    else:
        glm_corrected_particle_losses = np.round((np.maximum(yAll - yhatAll, 0)), 2)

    print(fitted_model.summary())
    print(settings.Config.glmImgQualiPredictors)
    return glm_corrected_particle_losses, iv_uAll, chichi


def apply_fit_to_all(originalDF, fitted_model):
    yAll = np.asarray(1 - (originalDF['matched_count'] / originalDF[
         'pre_count']))  # take observed losses of all wafers (not just water)

    yhatAll = np.asarray(1 / (1 + np.exp(
            (fitted_model.params[0] * originalDF[settings.Config.glmNpredictor] +\
             fitted_model.params[1] * originalDF[settings.Config.glmImgQualiPredictors[0]] +\
             fitted_model.params[2] * originalDF[settings.Config.glmImgQualiPredictors[1]] +\
             # fitted_model.params[3] +\
             0))))  # get fitted losses for all wafers
    return yAll, yhatAll


# def predictor_testing(df4glm, originalDF):
#     settings.Config.glmPredictorTesting = False
#     for f0 in np.arange(0, len(settings.Ndict)):
#         settings.Config.glmNpredictor = f0
#         for f1 in np.arange(0, len(settings.BDIdict)):
#             settings.Config.glmImgQualiPredictors[0] = settings.BDIdict[f1]
#             for f2 in np.arange(0, len(settings.BDIdict)):
#                 settings.Config.glmImgQualiPredictors[1] = settings.BDIdict[f2]
#                 glm_from_arrays(df4glm, originalDF)
#

def glm_from_arrays(df4glm, originalDF):

    global paramsDF
    paramsDF = pd.DataFrame()
    # for f0 in np.arange(0, len(settings.Ndict)):
    #     settings.Config.glmNpredictor = settings.Ndict[f0]
    #     for f1 in np.arange(0, len(settings.BDIdict)):
    #         settings.Config.glmImgQualiPredictors[0] = settings.BDIdict[f1]
    #         for f2 in np.arange(0, len(settings.BDIdict)):
    #             settings.Config.glmImgQualiPredictors[1] = settings.BDIdict[f2]
    endog, exog = make_glm_arrays(df4glm)
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    yAll, yhatAll = apply_fit_to_all(originalDF, glm_binom)
    glm_corrected_particle_losses, iv_uAll, chichi = glm_residuals(glm_binom, endog, yAll, yhatAll)

    params = glm_binom.params
    paramsDF = paramsDF.append([[settings.Config.glmNpredictor,
                                 settings.Config.glmImgQualiPredictors[0],
                                 settings.Config.glmImgQualiPredictors[1],
                                 params, glm_binom.deviance,
                                 glm_binom.aic, chichi[0], chichi[1]]])
    paramsDF.columns = ['Npredictor', 'ImgQualiPredictor1', 'ImgQualiPredictor2',
                        'coeffs', 'glmDeviance', 'glmAIC', 'glmChiSqStats', 'glmChiSqPvalue']
    return glm_corrected_particle_losses, yhatAll, iv_uAll, glm_binom


def glm_from_df(wafer_results, formula='SUCCESS ~ pre_count + BDI'):
    df4glm = make_glm_dataframe(wafer_results)
    glm_binom = smf.glm(formula=formula, data=df4glm,
                        family=sm.families.Binomial()).fit()

    endog, _ = make_glm_arrays(wafer_results)
    glm_corrected_particle_losses, yhatAll, iv_uAll = glm_residuals(glm_binom, endog)
    return glm_corrected_particle_losses, yhatAll, iv_uAll, glm_binom


def count_loss_glm(wafer_results):
    # wafer_results.sort_values(by=['treatment'], inplace=True)
    df4glm = prepare_df4glm(wafer_results)

    if settings.Config.countGLMformula is None:
        glm_corrected_particle_losses,\
        particle_losses_by_predictors,\
        particle_losses_by_predictors_upperInterval, \
        glm_binom =\
            glm_from_arrays(df4glm, wafer_results)
    else:  # TODO: the formula based method is not ready. So if it's needed it should be completed first analog to the array based glm.
        glm_corrected_particle_losses, \
        particle_losses_by_predictors, \
        particle_losses_by_predictors_upperInterval, \
        glm_binom =\
            glm_from_df(wafer_results, settings.Config.countGLMformula)


    glm_corrected_particle_losses_df = pd.DataFrame._from_arrays([glm_corrected_particle_losses,
                                                                  particle_losses_by_predictors,
                                                                  particle_losses_by_predictors_upperInterval],
                                                                 columns=['particle_loss_glmCorrected',
                                                                          'confounder_caused_particle_loss',
                                                                          'upperPredIntv_confounder_caused_particle_loss'],
                                                                 index=wafer_results.index)

    if not settings.Config.manualBDI:
        glm_corrected_particle_losses_df['BDI'] = round(1 / (glm_binom.params[1] * wafer_results[settings.Config.glmImgQualiPredictors[0]] +\
                                                  glm_binom.params[2] * wafer_results[settings.Config.glmImgQualiPredictors[1]]), 2)
        if glm_corrected_particle_losses_df['BDI'].max() < 0:
            glm_corrected_particle_losses_df['BDI'] = glm_corrected_particle_losses_df['BDI'] * -1  # convert to positive BDI range
        glm_corrected_particle_losses_df['IQI'] = glm_corrected_particle_losses_df['BDI'].max() -\
                                                  glm_corrected_particle_losses_df['BDI']

    wafer_results = pd.concat([wafer_results, glm_corrected_particle_losses_df], axis=1)
    return wafer_results, paramsDF


def area_change_glm(wafer_results, pam):
    # wafer_results['area_change_glmCorrected'] = wafer_results['area_change']   # for now just add same column. We can add a glm correction for area later if we need it

    pam = pd.merge(pam, wafer_results, on=['wafer', 'polymer', 'treatment'], how='left').dropna()
    paw = pam.loc[(pam['prop'] == 'area') & (pam['treatment'] == 'water')]
    paa = pam.loc[pam['prop'] == 'area']
    # endog = np.asarray(paw['change'])
    # exog = np.asarray(paw.loc[:, [settings.Config.glmNpredictor,
    #                               settings.Config.glmImgQualiPredictors[0],
    #                               settings.Config.glmImgQualiPredictors[1]
    #                               ]])
    # # exog = sm.add_constant(exog, prepend=False)
    # glm_gauss = sm.GLM(endog, exog, family=sm.families.Gaussian())

    formula = f'change ~ {settings.Config.glmNpredictor} +' \
              f' {settings.Config.glmImgQualiPredictors[0]} +' \
              f' {settings.Config.glmImgQualiPredictors[1]} -1'

    glm_gauss = smf.glm(formula=formula, data=paw, family=sm.families.Gaussian())


    fitted_gauss = glm_gauss.fit()

    yAll = np.asarray(paa['change'])
    yhatAll = np.asarray(fitted_gauss.params[0] * paa[settings.Config.glmNpredictor] +\
                         fitted_gauss.params[1] * paa[settings.Config.glmImgQualiPredictors[0]] +\
                         fitted_gauss.params[2] * paa[settings.Config.glmImgQualiPredictors[1]] +\
                         # fitted_gauss.params[3] +\
                         0)
    glm_corrected_area_changes, \
    area_changes_by_predictors_upperInterval, \
    chichi = glm_residuals(fitted_gauss, np.asarray(paw['change']), yAll, yhatAll)

    glm_corrected_area_changes_df = pd.DataFrame._from_arrays([glm_corrected_area_changes,
                                                                  yhatAll,
                                                                  area_changes_by_predictors_upperInterval],
                                                                 columns=['area_change_glmCorrected',
                                                                          'confounder_caused_area_change',
                                                                          'upperPredIntv_confounder_caused_area_change'],
                                                                 index=paa.index
                                                              )
    par = pd.concat([paa, glm_corrected_area_changes_df], axis=1)

    area_wafer_groups = par.groupby('wafer')
    for w, waf in area_wafer_groups:  # cycle through polymer groups
        area_change_glmCorrected_mean = waf.area_change_glmCorrected.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'area_change_glmCorrected'] = round(area_change_glmCorrected_mean, 2)
        confounder_caused_area_change_mean = waf.confounder_caused_area_change.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'confounder_caused_area_change'] = round(confounder_caused_area_change_mean, 2)
        upperPredIntv_confounder_caused_area_change_mean = waf.upperPredIntv_confounder_caused_area_change.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'upperPredIntv_confounder_caused_area_change'] = round(
        upperPredIntv_confounder_caused_area_change_mean, 2)

    return wafer_results
