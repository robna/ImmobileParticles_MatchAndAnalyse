import pandas as pd

BDIdict = {
    0: 'pre_histBGpeak',
    1: 'pre_histFGpeak',
    2: 'post_histBGpeak',
    3: 'post_histFGpeak',
    4: 'pre_histDelta',
    5: 'post_histDelta',
    6: 'histDeltaDiff',
    7: 'histBGpeakSum',
    8: 'histBGpeakDist',

}

Ndict = {
    0: 'pre_count',
    1: 'matched_count',
    2: 'post_count'
}

drops = {
    'wafers': ['', ''],  # a list of wafer names to be dropped from analysis
    'combis': [],  #['KOH', 'PET', 'PP'], ['', '']],  # alternatively to wafers lists with treatment and polymer can be used to drop these combinations (each list may include arbitrary numbers of polymers and treatments in arbitrary order)
    'polymers': [''],  # ['PS', 'PMMA', 'PVC', 'ABS'],  # a list of polymers to be dropped from analysis (all treatments)
    'treatments': ['', '']  # a list of treatments to be dropped from analysis (all polymers)
}


class Config:
    """
    Use this struct to specify parameters for data analysis
    """
    semiMelted: bool = False  # semiMelted wafer df was used before glm corrections were included, now use doubleMelt

    minParameterForBDI: float = 0  # smallest number for alpha and beta in BDI optimisation (only used for manual BDI)
    maxParameterForBDI: float = 12  # largest number for alpha and beta in BDI optimisation (only used for manual BDI)
    stepParameterForBDI: float = 1.0  # step size for alpha and beta in BDI optimisation (only used for manual BDI)
    manualBDI: bool = True  # True means running BDI module for creating BDI, False takes values from relevant GLM

    glmNpredictor: str = Ndict[0]  # what to use in GLM as a n-related predictor: one value of Ndict
    glmImgQualiPredictors: list = [BDIdict[4], BDIdict[8]] # 2 columns from wafer_results DF to be used as predictors, can be any of BDIdict

    formulaBasedGLM: bool = True
    countGLMformula: str = f'failure + success ~ {glmNpredictor} + ' \
                           f'{glmImgQualiPredictors[0]}  ' \
                           # f'{glmImgQualiPredictors[1]}'
    areaGLMformula: str = f'area_change ~ {glmNpredictor} + ' \
                          f'{glmImgQualiPredictors[1]}  ' \
                          # f'{glmImgQualiPredictors[1]} '

    glmOnTheWater: bool = True
    glmPredictorTesting: bool = False  # whether to test different input parameters for what make the best GLM model
    glmPlots: bool = True  # show GLM results as matplotlib figures
    useGLMpredIv: bool = False  # correct values with GLM fit based on upper limit of prediction interval (True) or fitted linearized regression (False)
    areaPerParticleGLM: bool = False


class reports:
    GLMparams = pd.DataFrame(index=[0])
