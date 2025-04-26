from pykrige.ok import OrdinaryKriging
from sklearn.base import BaseEstimator, RegressorMixin


class OrdinaryKrigingInterface(BaseEstimator, RegressorMixin):
    """
    Interface for using ordinary kriging with scikit-learn compatibility.

    Attributes:
        variogram_model (str): Variogram model to be used.
        verbose (bool): Determines if detailed messages will be displayed.
        enable_plotting (bool): Determines if plots will be generated during fitting.
        ok (OrdinaryKriging): Instance of the ordinary kriging model.
    """

    def __init__(self, variogram_model='linear', verbose=False, enable_plotting=False):
        """
        Initializes the ordinary kriging interface with the provided parameters.

        Args:
            variogram_model (str): Variogram model to be used (default: 'linear').
            verbose (bool): Determines if detailed messages will be displayed (default: False).
            enable_plotting (bool): Determines if plots will be generated during fitting (default: False).
        """

        self.variogram_model = variogram_model
        self.verbose = verbose
        self.enable_plotting = enable_plotting
        self.ok = None


    def fit(self, X, y):
        """
        Fits the ordinary kriging model to the provided data.

        Args:
            X (np.ndarray): Input coordinates (2D array with columns for X and Y).
            y (np.ndarray): Target values corresponding to the coordinates.

        Returns:
            self: Fitted model instance.
        """

        self.ok = OrdinaryKriging(
            X[:, 0], X[:, 1], y,
            variogram_model = self.variogram_model,
            verbose = self.verbose,
            enable_plotting = self.enable_plotting
        )

        return self


    def predict(self, X):
        """
        Makes predictions using the fitted ordinary kriging model.

        Args:
            X (np.ndarray): Input coordinates for prediction (2D array with columns for X and Y).

        Returns:
            np.ndarray: Predicted values for the provided coordinates.
        """

        z, ss = self.ok.execute('points', X[:, 0], X[:, 1])

        return z
