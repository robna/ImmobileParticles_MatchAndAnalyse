import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import settings
from matplotlib import pyplot as plt
from textwrap import wrap
from statsmodels.graphics.api import abline_plot


def paramsReport(fm, chichi):
    paramsDF = pd.DataFrame(pd.DataFrame([fm.params, fm.pvalues], index=['coeff', 'p']).unstack()).T
    paramsDF.columns = [' '.join(col).strip() for col in paramsDF.columns.values]  # turn multilevel (hierachical) columns into flat columns by combining and repeating column name parts
    paramsDF[['deviance', 'aic', 'reduction', 'pMean', 'pMax', 'family', 'summary']] = [fm.deviance, fm.aic, np.nan,
                                                                                fm.pvalues.mean(), fm.pvalues.max(),
                                                                                type(fm.family).__name__, fm.summary()]
    paramsDF.index = [settings.reports.GLMparams.index[-1]+1]
    settings.reports.GLMparams = settings.reports.GLMparams.append(paramsDF)


def glmControlPlots(fitted_model, y, yAll, yhat, yhatAll, line_fit, meanRespAll, rr, rp, iv_u, iv_l, iv_uAll, iv_lAll, rrivU, rrivL):

    # df = pd.DataFrame({'yAll': yAll, 'yhatAll': yhatAll,
    #                    'meanRespAll': meanRespAll, 'rr': rr,
    #                    'iv_uAll': iv_uAll, 'iv_lAll': iv_lAll,
    #                    'rrivU': rrivU, 'rrivL': rrivL})
    # df.sort_values(by='yhatAll', inplace=True)

    ivs_sorted = np.column_stack(
        [yhatAll.T, iv_uAll.T, iv_lAll.T, yAll.T, meanRespAll.T, rrivU, rrivL])  # combine fitted losses with predict ints for sorting
    ivs_sorteds = np.column_stack(
        [yhat.T, iv_u.T, iv_l.T])  # also show original (non-extrapolated) upper pred interval for comparison
    ivs_sorted = ivs_sorted[
        ivs_sorted[:, 0].argsort()]  # sort predict int values by fitted losses to avoid distorted line plot
    ivs_sorteds = ivs_sorteds[ivs_sorteds[:, 0].argsort()]

    fig, ax = plt.subplots(2, 2, figsize=[20, 20])
    # ax.fill_between(yhat, iv_u, iv_l, color="#b9cfe7", edgecolor="")  # attempt to plot predict int as area
    ax[0, 0].plot(ivs_sorted[:, 0], ivs_sorted[:, 1], color='lightgray', linestyle='dashed')  # plot upper predict int
    ax[0, 0].plot(ivs_sorteds[:, 0], ivs_sorteds[:, 1], color='red')
    ax[0, 0].plot(ivs_sorted[:, 0], ivs_sorted[:, 2], color='lightgray', linestyle='dashed')  # plot lower predict int
    ax[0, 0].plot(ivs_sorteds[:, 0], ivs_sorteds[:, 2], color='red')
    abline_plot(model_results=line_fit, ax=ax[0, 0], color='black')  # plot linear regression line
    # ax[0, 0].scatter(yhatAll, meanRespAll, s=20, color='black')
    # ax[0, 0].scatter(yhat, line_fit.predict(sm.add_constant(yhat)), s=20, color='red', zorder=8)
    # ax[0, 0].bar(ivs_sorted[:, 0], rr, bottom=meanRespAll, color='black', width=0.002)
    ax[0, 0].scatter(yhatAll, yAll, s=50)  # plot all original loss values against predicted loss values
    ax[0, 0].scatter(yhat, y, s=10)  # plot water point in different color on top
    ax[0, 0].set_xlabel('fitted relative changes (attributed to n and image quality)')
    ax[0, 0].set_ylabel('observed relative changes (blue dots, water with orange kernel)')
    ax[0, 0].set_title('Linearized model')
    ax[0, 0].set_ylim(yAll.min() - (yAll.max()-yAll.min()) * 0.1, yAll.max() + (yAll.max()-yAll.min()) * 0.1)
    ax[0, 0].set_xlim(yhatAll.min() - (yhatAll.max()-yhatAll.min()) * 0.1, yhatAll.max() + (yhatAll.max()-yhatAll.min()) * 0.1)

    ax[0, 1].scatter(yhat, rp, s=50, color='C2')
    # ax[0, 1].scatter(meanRespAll, rr, s=50, color='C2')
    # ax[0, 1].scatter(ivs_sorted[:, 0], rr, s=50, color='C2')
    # ax[0, 1].plot(ivs_sorted[:, 0], ivs_sorted[:, 5], color='black')
    # ax[0, 1].scatter(np.arange(1, len(yAll) + 1, 1), iv_lAll, s=50, color='black')
    # ax[0, 1].bar(yhatAll, rr, color='C2', width=0.01)
    # ax[0, 1].bar(meanRespAll, rr, color='C2', width=0.005)
    # ax[0, 1].bar(ivs_sorted[:, 0], rr, color='C2', width=0.006)
    ax[0, 1].hlines(0, min(yhat), max(yhat))
    # ax[0, 1].hlines(0, 0, len(yAll))
    ax[0, 1].set_xlabel('fitted values')
    # ax[0, 1].set_ylabel('Standard response residuals (corrected relative changes)')
    ax[0, 1].set_ylabel('Pearson residuals')
    # ax[0, 1].set_title('Relative changes caused by treatment after GLM correction')
    ax[0, 1].set_title('Residual Dependence Plot')

    ax[1, 0].scatter(np.arange(1, len(yAll) + 1, 1), ivs_sorted[:, 3])
    ax[1, 0].scatter(np.arange(1, len(yAll) + 1, 1), ivs_sorted[:, 0], s=20, color='grey')
    ax[1, 0].plot(np.arange(1, len(yAll) + 1, 1), ivs_sorted[:, 4], color='black')
    # ax[1, 0].set_xlabel('id')
    ax[1, 0].axes.xaxis.set_visible(False)
    ax[1, 0].set_ylabel('observed and fitted relative changes')
    ax[1, 0].set_title(
        "\n".join(wrap('Original relative change (blue) and change attributed to n and image quality (i.e. fitted values, grey), Black line shows the fitted mean', 80)))

    ax[1, 1].bar(np.arange(1, len(yAll) + 1, 1), yAll)  # , width=0.2)
    # ax[1, 1].bar(np.arange(1, len(yAll) + 1, 1), rr, color='C2')  # , width=0.2)
    ax[1, 1].scatter(np.arange(1, len(yAll) + 1, 1), rr, color='black', zorder=4)
    ax[1, 1].bar(np.arange(1, len(yAll) + 1, 1), -yhatAll, bottom=yAll, color='C1', alpha=0.5)  # , width=0.2)
    ax[1, 1].set_ylim(0, np.max([1, max(yAll)]))
    # ax[1, 1].set_xlabel('id')
    ax[1, 1].axes.xaxis.set_visible(False)
    ax[1, 1].set_ylabel('relative changes')
    ax[1, 1].set_title("\n".join(wrap(
        'Relative changes: original (tops), attributed to n and image quality (upper bar) and attributed to treatment effects (bottom bar)',
        80)))

    textstr1 = f'Family: {type(fitted_model.family).__name__},    Link: {type(fitted_model.family.link).__name__}  \nFormula: {fitted_model.model.formula} \nDeviance: {fitted_model.deviance.round(2)}'
    textstr2 = f'                       Coefficients \n{fitted_model.params.round(4).to_string()}'
    textstr3 = f'                       p-Values \n{fitted_model.pvalues.round(3).to_string()}'
    plt.gcf().text(0.05, 0.92, textstr1, fontsize=16)
    plt.gcf().text(0.6, 0.92, textstr2, fontsize=14)
    plt.gcf().text(0.8, 0.92, textstr3, fontsize=14)
    fig.show()
    plt.show(block=True)
    return


def prepare_df4glm(wafer_results):
    df4glm = wafer_results.copy()
    df4glm['success'] = df4glm['matched_count']
    df4glm['failure'] = df4glm.pre_count - df4glm.matched_count
    return df4glm


def make_glm_arrays(df4glm):
    endog = np.asarray(df4glm.loc[df4glm.treatment == 'water', ['success', 'failure']])  if settings.Config.glmOnTheWater else np.asarray(df4glm['success', 'failure'])# take only water for fitting the glm
    exog = np.asarray(df4glm.loc[df4glm.treatment == 'water', [settings.Config.glmNpredictor,
                              settings.Config.glmImgQualiPredictors[0],
                              settings.Config.glmImgQualiPredictors[1]]]) if settings.Config.glmOnTheWater else np.asarray(df4glm[settings.Config.glmNpredictor, settings.Config.glmImgQualiPredictors[0], settings.Config.glmImgQualiPredictors[1]])  # take only water for fitting the glm
    exogAll = np.asarray(df4glm.loc[:, [settings.Config.glmNpredictor,
                                        settings.Config.glmImgQualiPredictors[0],
                                        settings.Config.glmImgQualiPredictors[1]
                                        ]])
    exog = sm.add_constant(exog, prepend=True)
    exogAll = sm.add_constant(exogAll, prepend=True)
    return endog, exog, exogAll


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

def chiTest(y, yhat, ddof=0):
    chichi = stats.chisquare(f_obs=y, f_exp=yhat, ddof=ddof)  # chi² seems not correct... what is correct input order? obs=yhat exp=y, or vice versa?
    # GLM internal Pearson Chi² (for 3 DOF, i.e. GLM with 3 predictors) needs to be above [7.815, 11.345, 16.266] for p [0.05, 0.01, 0.001]
    return chichi


def glm_residuals(fitted_model, endog, yAll, yhatAll):
    if endog.ndim > 1:
        y = 1 - (endog[:, 0] / endog.sum(1))  # loss rate = 1 - matched / pre
        yhat = 1 - fitted_model.mu  # fitted loss values for water wafers  TODO: is this correct to take 1 - mu?
        rr = np.maximum(yAll - yhatAll, 0)  # residual response for all data points (= glm corrected rates, when using prediction mean)
    else:
        y = endog
        yhat = fitted_model.mu
        rr = yAll - yhatAll  # residual response for all data points (= glm corrected rates, when using prediction mean)
        rr = np.maximum(yAll - yhatAll, 0)

    chichi = chiTest(y, yhat, ddof=0)
    rp = fitted_model.resid_pearson

    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()  # make linear regression between original and fitted values
    meanRespAll = line_fit.predict(sm.add_constant(yhatAll))
    _, iv_l, iv_u = wls_prediction_std(line_fit)  # calculate predict intervals from linear regression
    iv_lAll = extrap(yhatAll, yhat, iv_l, ord=3)  # get lower pred interval for all point by interpolation / extrapolation
    iv_uAll = extrap(yhatAll, yhat, iv_u, ord=3)  # get upper pred interval for all point by interpolation / extrapolation

    rrivU = iv_uAll - yhatAll  # residual response upper prediction interval
    # rrivU = iv_uAll - meanRespAll  # residual response upper prediction interval
    rrivL = iv_lAll - yhatAll  # residual response lower prediction interval
    # rrivL = iv_lAll - meanRespAll  # residual response lower prediction interval
    if settings.Config.useGLMpredIv:
        rr[(rr <= rrivU) & (rr >= rrivL)] = 0
        rr[rr > rrivU] = rr[rr > rrivU] - rrivU[rr > rrivU]
        rr[rr < rrivL] = rr[rr < rrivL] - rrivL[rr < rrivL]
        # rr[rr > 0] = np.maximum(rr[rr > 0] - rrivU[rr > 0], 0)  # adjust positive rates by upper interval boundary
        # rr[rr < 0] = np.minimum(rr[rr < 0] - rrivL[rr < 0], 0)  # adjust negative rates by lower interval boundary

    # print(f'Predictors:   {settings.Config.glmNpredictor}   ,   ' \
    #       f'{settings.Config.glmImgQualiPredictors[0]}   and   {settings.Config.glmImgQualiPredictors[1]}')

    if not settings.Config.glmPredictorTesting:
        print(fitted_model.summary2())
    else:
        print(fitted_model.params)

    if settings.Config.glmPlots and not settings.Config.glmPredictorTesting:
        glmControlPlots(fitted_model, y, yAll, yhat, yhatAll, line_fit, meanRespAll, rr, rp, iv_u, iv_l, iv_uAll, iv_lAll,
                        rrivU,  # if settings.Config.useGLMpredIv else None,
                        rrivL)  # if settings.Config.useGLMpredIv else None)

    return rr, rrivU, rrivL, chichi


def glm_from_arrays(df4glm, originalDF):
    # for f0 in np.arange(0, len(settings.Ndict)):
    #     settings.Config.glmNpredictor = settings.Ndict[f0]
    #     for f1 in np.arange(0, len(settings.BDIdict)):
    #         settings.Config.glmImgQualiPredictors[0] = settings.BDIdict[f1]
    #         for f2 in np.arange(0, len(settings.BDIdict)):
    #             settings.Config.glmImgQualiPredictors[1] = settings.BDIdict[f2]

    endog, exog, exogAll = make_glm_arrays(df4glm)
    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    yAll = np.asarray(originalDF['particle_loss'])  # take observed losses of all wafers (not just water)
    # yhatAll = np.asarray(1 / (1 + np.exp(
    #     (glm_binom.params[1] * originalDF[settings.Config.glmNpredictor] + \
    #      glm_binom.params[2] * originalDF[settings.Config.glmImgQualiPredictors[0]] + \
    #      glm_binom.params[3] * originalDF[settings.Config.glmImgQualiPredictors[1]] + \
    #      glm_binom.params[0] + \
    #      0))))  # get fitted losses for all wafers
    yhatAll = 1 - glm_binom.predict(exogAll)  # alternative to the manual way above

    glm_corrected_particle_losses, iv_uAll, iv_lAll, chichi = glm_residuals(glm_binom, endog, yAll, yhatAll)

    paramsReport(glm_binom, chichi)
    return glm_corrected_particle_losses, yhatAll, iv_uAll, iv_lAll, glm_binom


def glm_from_df(wafer_results):
    endogAll = np.asarray(wafer_results.particle_loss)  # = np.asarray(1 - df4glm['success'] / (df4glm['success'] + df4glm['failure']))
    df4glmAll = prepare_df4glm(wafer_results)
    df4glm = df4glmAll.loc[df4glmAll.treatment == 'water'] if settings.Config.glmOnTheWater else df4glmAll
    endog = np.asarray(df4glm.particle_loss)  # = np.asarray(1 - df4glm['success'] / (df4glm['success'] + df4glm['failure']))
    # formula = f'failure + success ~ {settings.Ndict[0]} + ' \
    #           f'{settings.BDIdict[4]}  ' \
    #          # f'{settings.BDIdict[6]}'

    # formula = f'failure + success ~ {settings.Config.glmNpredictor} + ' \
    #           f'{settings.Config.glmImgQualiPredictors[0]} + ' \
    #           f'{settings.Config.glmImgQualiPredictors[1]}'  # with "-1" in the end to run without constant term

    glm_binom = smf.glm(formula=settings.Config.countGLMformula, data=df4glm,
                        family=sm.families.Binomial()).fit()
    # yhatAll = np.asarray(1 / (1 + np.exp(
    #     (glm_binom.params[1] * wafer_results[settings.Config.glmNpredictor] + \
    #      glm_binom.params[2] * wafer_results[settings.Config.glmImgQualiPredictors[0]] + \
    #      glm_binom.params[3] * wafer_results[settings.Config.glmImgQualiPredictors[1]] + \
    #      glm_binom.params[0] + \
    #      0))))  # get fitted losses for all wafers
    yhatAll = np.asarray(glm_binom.predict(wafer_results))  # alternative to the manual way above
    glm_corrected_particle_losses, rrivU, rrivL, chichi = glm_residuals(glm_binom, endog, endogAll, yhatAll)

    paramsReport(glm_binom, chichi)
    return glm_corrected_particle_losses, yhatAll, rrivU, rrivL, glm_binom


def count_loss_glm(wafer_results):
    # wafer_results.sort_values(by=['treatment'], inplace=True)
    df4glm = prepare_df4glm(wafer_results)

    if settings.Config.formulaBasedGLM:
        glm_corrected_particle_losses, \
        particle_losses_by_predictors, \
        particle_losses_by_predictors_upperInterval, \
        particle_losses_by_predictors_lowerInterval, \
        glm_binom = \
            glm_from_df(wafer_results)
    else:
        glm_corrected_particle_losses, \
        particle_losses_by_predictors, \
        particle_losses_by_predictors_upperInterval, \
        particle_losses_by_predictors_lowerInterval, \
        glm_binom = \
            glm_from_arrays(df4glm, wafer_results)

    glm_corrected_particle_losses_df = pd.DataFrame._from_arrays([glm_corrected_particle_losses,
                                                                  particle_losses_by_predictors,
                                                                  particle_losses_by_predictors_upperInterval,
                                                                  particle_losses_by_predictors_lowerInterval],
                                                                 columns=['particle_loss_glm_corrected',
                                                                          'confounder_caused_particle_loss',
                                                                          'upperPredIntv_confounder_caused_particle_loss',
                                                                          'lowerPredIntv_confounder_caused_particle_loss'],
                                                                 index=wafer_results.index)

    if not settings.Config.manualBDI:
        glm_corrected_particle_losses_df['BDI'] = round(1 / (1 + np.exp((
                glm_binom.params[-2] * wafer_results[settings.Config.glmImgQualiPredictors[0]]
            # +  glm_binom.params[-1] * wafer_results[settings.Config.glmImgQualiPredictors[1]]
        ))), 3)
        if glm_corrected_particle_losses_df['BDI'].max() < 0:
            glm_corrected_particle_losses_df['BDI'] -= glm_corrected_particle_losses_df['BDI'].min()  # convert to positive BDI range
        glm_corrected_particle_losses_df['IQI'] = glm_corrected_particle_losses_df['BDI'].max() - \
                                                  glm_corrected_particle_losses_df['BDI']

    reduction = abs(glm_corrected_particle_losses_df.particle_loss_glm_corrected).sum() / abs(wafer_results.particle_loss).sum() - 1
    settings.reports.GLMparams.loc[settings.reports.GLMparams.index[-1], 'reduction'] = reduction
    # plt.gcf().text(0.65, 0.08, f'Reduction:   {reduction}', fontsize=16)
    print(f'Reduction:   {reduction}')
    try:
        wafer_results = pd.concat([wafer_results, glm_corrected_particle_losses_df], axis=1, verify_integrity=True)
    except ValueError:
        pass

    return wafer_results


def area_change_glm(wafer_results, pam):
    if settings.Config.areaPerParticleGLM:
        wafer_results = area_change_RepeatedMeasuresGlm(wafer_results, pam)
    else:
        wafer_results_AbsChange = wafer_results.copy()
        wafer_results_AbsChange.area_change = abs(wafer_results.area_change)
        ww = wafer_results_AbsChange.loc[wafer_results_AbsChange.treatment == 'water'] if \
            settings.Config.glmOnTheWater else wafer_results_AbsChange  # get water wafers

        # formula = f'area_change ~ {settings.Ndict[0]} + ' \
        #           f'{settings.BDIdict[8]} ' \
        #           # f'{settings.BDIdict[2]}'  # with "-1" in the end to run without constant term

        # formula = f'area_change ~ {settings.Config.glmNpredictor} + ' \
        #           f'{settings.Config.glmImgQualiPredictors[0]} + ' \
        #           f'{settings.Config.glmImgQualiPredictors[1]}'  # with "-1" in the end to run without constant term

        link_g = sm.genmod.families.links.identity
        glm_gauss = smf.glm(formula=settings.Config.areaGLMformula, data=ww, family=sm.families.Gamma(link_g()))
        fitted_gauss = glm_gauss.fit()

        yAll = np.asarray(wafer_results_AbsChange['area_change'])
        # yhatAll = np.asarray(fitted_gauss.params[1] * wafer_results_AbsChange[settings.Config.glmNpredictor] +\
        #                      fitted_gauss.params[2] * wafer_results_AbsChange[settings.Config.glmImgQualiPredictors[0]] +\
        #                      fitted_gauss.params[3] * wafer_results_AbsChange[settings.Config.glmImgQualiPredictors[1]] +\
        #                      fitted_gauss.params[0] +\
        #                      0)
        yhatAll = fitted_gauss.predict(wafer_results_AbsChange)  # this returns the same as the manual prediction above...

        glm_corrected_area_changes, \
        area_changes_by_predictors_upperInterval, \
        area_changes_by_predictors_lowerInterval, \
        chichi = glm_residuals(fitted_gauss, np.asarray(ww['area_change']), yAll, yhatAll)

        gc_ac = pd.DataFrame._from_arrays([glm_corrected_area_changes,  # gc_ac = glm_corrected_area_changes_df
                                           yhatAll,
                                           area_changes_by_predictors_upperInterval,
                                           area_changes_by_predictors_lowerInterval],
                                          columns=['area_change_glm_corrected',
                                                   'confounder_caused_area_change',
                                                   'upperPredIntv_confounder_caused_area_change',
                                                   'lowerPredIntv_confounder_caused_area_change'],
                                          index=wafer_results.index)
        gc_ac.loc[wafer_results.area_change < 0, 'area_change_glm_corrected'] *= -1  # reverse absolutation
        gc_ac.loc[wafer_results.area_change < 0, 'confounder_caused_area_change'] *= -1  # reverse absolutation
        try:  # this error exception is needed to run the glm parameter testing loops
            wafer_results = pd.concat([wafer_results, gc_ac], axis=1, verify_integrity=True)
        except ValueError:
            pass
            # wafer_results['area_change_glm_corrected'] = wafer_results['area_change']   # activate this instead of the glm, to run without area glm correction
        paramsReport(fitted_gauss, chichi)
        reduction = abs(gc_ac.area_change_glm_corrected).sum() / abs(wafer_results.area_change).sum() - 1
        settings.reports.GLMparams.loc[settings.reports.GLMparams.index[-1], 'reduction'] = reduction
        # plt.gcf().text(0.65, 0.08, f'Reduction:   {reduction}', fontsize=16)
        print(f'Reduction:   {reduction}')
    return wafer_results


def area_change_RepeatedMeasuresGlm(wafer_results, pam):  # TODO: taking absolute area changes is not yet implemented in single particle based area glm
    # wafer_results['area_change_glm_corrected'] = wafer_results['area_change']   # for now just add same column. We can add a glm correction for area later if we need it

    pam = pd.merge(pam, wafer_results, on=['wafer', 'polymer', 'treatment'], how='left').dropna()
    paa = pam.loc[pam['prop'] == 'area']

    paa_AbsChange = paa.copy()
    paa_AbsChange.change = abs(paa_AbsChange.change)

    paw = paa_AbsChange.loc[paa_AbsChange['treatment'] == 'water'] if settings.Config.glmOnTheWater else paa_AbsChange

    formula = f'change ~ {settings.Config.glmNpredictor} +' \
              f' {settings.Config.glmImgQualiPredictors[1]} ' \
              # f' {settings.Config.glmImgQualiPredictors[1]}'  # with "-1" in the end to run without constant term

    link_g = sm.genmod.families.links.identity
    glm_gauss = smf.glm(formula=formula, data=paw, family=sm.families.Gamma(link_g()))

    fitted_gauss = glm_gauss.fit()

    yAll = np.asarray(paa['change'])
    # yhatAll = np.asarray(fitted_gauss.params[1] * paa[settings.Config.glmNpredictor] +\
    #                      fitted_gauss.params[2] * paa[settings.Config.glmImgQualiPredictors[0]] +\
    #                      fitted_gauss.params[3] * paa[settings.Config.glmImgQualiPredictors[1]] +\
    #                      fitted_gauss.params[0] +\
    #                      0)
    yhatAll = fitted_gauss.predict(paa)  # this returns the same as the manual prediction above...

    glm_corrected_area_changes, \
    area_changes_by_predictors_upperInterval, \
    area_changes_by_predictors_lowerInterval, \
    chichi = glm_residuals(fitted_gauss, np.asarray(paw['change']), yAll, yhatAll)

    gc_ac = pd.DataFrame._from_arrays([glm_corrected_area_changes,
                                       yhatAll,
                                       area_changes_by_predictors_upperInterval,
                                       area_changes_by_predictors_lowerInterval],
                                      columns=['area_change_glm_corrected',
                                               'confounder_caused_area_change',
                                               'upperPredIntv_confounder_caused_area_change',
                                               'lowerPredIntv_confounder_caused_area_change'],
                                      index=paa.index)
    gc_ac.loc[paa.change < 0, 'area_change_glm_corrected'] *= -1  # reverse absolutation
    gc_ac.loc[paa.change < 0, 'confounder_caused_area_change'] *= -1  # reverse absolutation
    par = pd.concat([paa, gc_ac], axis=1)

    area_wafer_groups = par.groupby('wafer')
    for w, waf in area_wafer_groups:  # cycle through polymer groups
        area_change_glm_corrected_mean = waf.area_change_glm_corrected.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'area_change_glm_corrected'] = area_change_glm_corrected_mean
        confounder_caused_area_change_mean = waf.confounder_caused_area_change.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'confounder_caused_area_change'] = confounder_caused_area_change_mean
        upperPredIntv_confounder_caused_area_change_mean = waf.upperPredIntv_confounder_caused_area_change.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'upperPredIntv_confounder_caused_area_change'] = upperPredIntv_confounder_caused_area_change_mean
        lowerPredIntv_confounder_caused_area_change_mean = waf.lowerPredIntv_confounder_caused_area_change.mean()
        wafer_results.loc[(wafer_results.wafer == w), 'lowerPredIntv_confounder_caused_area_change'] = lowerPredIntv_confounder_caused_area_change_mean

    paramsReport(fitted_gauss, chichi)
    settings.reports.GLMparams.loc[settings.reports.GLMparams.index[-1], 'reduction'] = \
        abs(gc_ac.area_change_glm_corrected).sum() / abs(wafer_results.area_change).sum() - 1
    return wafer_results
