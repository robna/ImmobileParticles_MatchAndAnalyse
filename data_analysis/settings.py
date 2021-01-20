BDIdict = {
    0: 'pre_histBGpeak',
    1: 'pre_histFGpeak',
    2: 'post_histBGpeak',
    3: 'post_histFGpeak',
    4: 'pre_histDelta',
    5: 'post_histDelta',
    6: 'histDeltaDiff',
    7: 'histBGpeakSum',
    8: 'histBGpeakDist'
}

Ndict = {
    0: 'pre_count',
    1: 'matched_count',
    2: 'post_count'
}

class Config:
    """
    Use this struct to specify parameters for data analysis
    """
    semiMelted: bool = False  # semiMelted wafer df was used before glm corrections were included, now use doubleMelt
    minParameterForBDI: float = -10  # smallest number for alpha and beta in BDI optimisation (only used for manual BDI)
    maxParameterForBDI: float = 11  # largest number for alpha and beta in BDI optimisation (only used for manual BDI)
    stepParameterForBDI: float = 1.0  # step size for alpha and beta in BDI optimisation (only used for manual BDI)
    manualBDI: bool = False  # True means running BDI module for creating BDI, False takes values from relevant GLM
    glmPredictorTesting: bool = True  # whether to test different input parameters for what make the best GLM model
    glmImgQualiPredictors: list = [BDIdict[4], BDIdict[5]]  # 2 columns from wafer_results DF to be used as predictors, can be any of BDIdict
    glmNpredictor: str = Ndict[1]  # what to use in GLM as a n-related predictor: one value of Ndict
    countGLMformula: str = None  # glm can be run by passing a custom formula, default (None) results in array based GLM
    areaGLMformula: str = None
    wafers2drop: list = []  # any wafers to be excluded from analysis TODO: not yet implemented in wafer_wrangling
    polymers2drop: list = []  # any polymers to be excluded from analysis
    treatments2drop: list = []  # any treatments to be excluded from analysis
    glmPlots: bool = False  # show GLM results as matplotlib figures
    useGLMpredIv: bool = True  # correct values with GLM fit based on upper limit of prediction interval (True) or fitted linearized regression (False)
