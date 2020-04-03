class Config:

    ## Parameters
    AREA_LIM = 2.0e-4
    PERIMETER_LIM = 1e-4
    ASPECT_RATIO_LIM = 5.0
    OCCUPATION_INTERVAL = (0.23, 0.90)
    COMPACTNESS_INTERVAL = (3e-3, 1e-1)
    SWT_TOTAL_COUNT = 10
    SWT_STD_LIM = 20.0
    STROKE_WIDTH_SIZE_RATIO_LIM = 0.02            ## Min value
    STROKE_WIDTH_VARIANCE_RATIO_LIM = 0.15        ## Min value
    STEP_LIMIT = 10
    KSIZE = 3
    ITERATION = 7
    MARGIN = 10