"""
This file contains constants and configurations used in the project.

Constants:
    SUITABILITY_FILEPATH (str): Path to the sps suitability file.
    ABUNDANCE_FILEPATH (str): Path to the sps abundance simulation file.
    P_1 (str): Path to the bioclimatic data file (variable 01).
    P_2 (str): Path to the bioclimatic data file (variable 12).

Columns:
    SKLEARN_X_COLUMNS (list): Input columns for scikit-learn models.
    SKLEARN_Y_COLUMN (str): Output column for scikit-learn models.
    REGRESSION_KRIGING_X_COLUMNS (list): Input columns for regression kriging.
    REGRESSION_KRIGING_P_COLUMN (list): Projection columns for regression kriging.
    REGRESSION_KRIGING_Y_COLUMN (str): Output column for regression kriging.

Others:
    PROJECTIONS_FOLDER (str): Path to the projections folder.
"""

EXAMPLE_SUITABILITY_FILEPATH = "sdswrapper/y/projection-Time0kyrBP-Replica1-Sample95.asc"
EXAMPLE_ABUNDANCE_FILEPATH = "sdswrapper/y/HW_simulated_population.pkl"
EXAMPLE_P_1 = "sdswrapper/p/wc2.1_2.5m_bio_1.tif"
EXAMPLE_P_2 = "sdswrapper/p/wc2.1_2.5m_bio_12.tif"

SKLEARN_X_COLUMNS = ['coordenada_X', 'coordenada_Y', 'bio01', 'bio12']
SKLEARN_Y_COLUMN = 'y'
REGRESSION_KRIGING_X_COLUMNS = ['coordenada_X', 'coordenada_Y']
REGRESSION_KRIGING_P_COLUMN = ['bio01', 'bio12']
REGRESSION_KRIGING_Y_COLUMN = 'y'

PROJECTIONS_FOLDER = 'x'