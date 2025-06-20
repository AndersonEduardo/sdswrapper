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

import os

EXAMPLE_SAMPLE_FILEPATH = os.path.join('sdswrapper','sample', 'S50_P50.xlsx')
EXAMPLE_FEATURES_FILEPATH = os.path.join('sdswrapper','features')
EXAMPLE_PROBABILITY_SURFACE_FILEPATH = os.path.join('sdswrapper','probability_surface', 
                                                    'probability_surface_example.asc')
