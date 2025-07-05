import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from pykrige.rk import RegressionKriging

from sdswrapper.ordinarykriginginterface import OrdinaryKrigingInterface



class Models:
    """
    Class for training and evaluating machine learning and kriging models.

    Attributes:
        models (list): List of models to be trained.
        X (pd.DataFrame): Input data for training.
        P (pd.DataFrame): Spatial projection data.
        XP (pd.DataFrame): Combination of X and P.
        Y (pd.Series): Output (target) data.
        k (int): Number of folds for cross-validation.
        projections_folder (str): Path to save projections.
        spatial_groups (np.ndarray): Spatial groups for cross-validation.
    """

    def __init__(self, models: list, X: pd.DataFrame, p: pd.DataFrame, y: pd.Series, 
                 k: int, projections_folder: str) -> None:
        """
        Initializes the Models class with the provided data and settings.

        Args:
            models (list): List of models to be trained.
            X (pd.DataFrame): Coordinate data. Shape should be (n_samples, 2 columns).
            p (pd.DataFrame): Feature data. Shape should be (n_samples, n_features).
            y (pd.Series): Target data. Shape should be (n_samples).
            k (int): Number of folds for cross-validation.
            projections_folder (str): Path to save projections.

        Raises:
            Exception: If the data contains infinite or NaN values.
        """

        if np.isinf(X).any().any():

            raise Exception("`X` contains infinite values.")

        if np.isinf(y).any():

            raise Exception("`y` contains infinite values.")

        if np.isinf(p).any().any():

            raise Exception("`p` contains infinite values.")

        if np.isnan(X).any().any():

            raise Exception("`X` contains infinite values.")

        if np.isnan(y).any():

            raise Exception("`y` contains infinite values.")

        if np.isnan(p).any().any():

            raise Exception("`p` contains infinite values.")


        self.models = models
        self.X = self.set_X(X)
        self.P = self.set_P(p)
        self.XP = self.set_XP(self.X, self.P)
        self.Y = self.set_Y(y)
        self.k = k
        self.sample_size = X.shape[0]
        self.projections_folder = projections_folder
        self.spatial_groups = KMeans(n_clusters=self.k, random_state=42).fit_predict(self.X)


    def __check_data(self, data: pd.DataFrame|pd.Series):
        """
        Checks if the data contains NaN or infinite values.

        Args:
            data (pd.DataFrame): Data to be checked.

        Returns:
            pd.DataFrame: Verified data.

        Raises:
            Exception: If the data contains NaN or infinite values.
        """

        # NaN
        if np.isnan(data).any().any():

            print()
            print('*data = ', data)
            print()

            raise Exception("`data` contains NaN values.")

        # Inf
        if np.isinf(data).any().any():

            print()
            print('*data = ', data)
            print()

            raise Exception("`data` contains infinite values.")

        return data


    def set_X(self, X: pd.DataFrame):
        """
        Sets the input data X after verification and conversion.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Processed input data.
        """

        if not isinstance(X, pd.DataFrame):

            raise Exception(f"`X` must be a pandas DataFrame. Found: {type(X)}")

        if X.empty:

            raise Exception("`X` cannot be an empty DataFrame.")


        X = X.reset_index(drop=True).astype('float32')

        X = self.__check_data(X)

        return X


    def set_P(self, p: pd.DataFrame):
        """
        Sets the projection data P after verification and conversion.

        Args:
            p (pd.DataFrame): Features data.

        Returns:
            pd.DataFrame: Processed features data.
        """

        if not isinstance(p, pd.DataFrame):

            raise Exception(f"`p` must be a pandas DataFrame. Found: {type(p)}")

        if p.empty:

            print('Warning: `p` is empty.')

        p = p.reset_index(drop=True).astype('float32')

        p = self.__check_data(p)

        return p


    def set_Y(self, y: pd.Series):
        """
        Sets the output data Y after verification and conversion.

        Args:
            y (pd.Series): Output data.

        Returns:
            pd.Series: Processed output data.
        """

        if not isinstance(y, pd.Series):

            raise Exception(f"`y` must be a pandas DataFrame. Found: {type(y)}")

        if y.empty:

            raise Exception("`y` cannot be an empty DataFrame.")

        # if y.shape[1] > 1:

        #     raise Exception("`y` must have only one column.")


        y = y.reset_index(drop=True).astype('float32')

        y = self.__check_data(y)

        return y


    def set_XP(self, X: pd.DataFrame, p: pd.DataFrame):
        """
        Combines the input data X and projection data P.

        Args:
            X (pd.DataFrame): Input data.
            p (pd.DataFrame): Projection data.

        Returns:
            pd.DataFrame: Combination of X and P.
        """

        if p.empty:

            print("Warning: `p` is empty. The property self.XP will be composed only by X. This can lead to problemns in the models that require `p`.")

            return X.astype('float32').copy()
        
        else:

            return pd.concat([X, p], axis=1).astype('float32').copy()


    def fit(self):
        """
        Trains the provided models using cross-validation.

        Returns:
            list: List of dictionaries containing metrics and trained models.
        """

        output = list()

        for name, model in self.models:

            model_type = None

            if isinstance(model, RegressionKriging):

                if self.P.empty:

                    print("Warning: `P` must be provided for Regression Kriging models. Skipping model:", name)

                    continue

                model_type = 'KR'

                model_metrics, model_trained = self._fit_regression_kriging(model)

            elif isinstance(model, OrdinaryKrigingInterface):

                model_type = 'KR'

                model_metrics, model_trained = self._fit_ordinary_kriging_models(model)

            else:

                if self.P.empty:

                    print("Warning: `P` must be provided for Regression Kriging models. Skipping model:", name)

                    continue

                model_type = 'SK'

                model_metrics, model_trained = self._fit_sklearn_models(model)

            output.append(
                {
                    'sample_size': self.sample_size, 
                    'name': name, 
                    'model_type': model_type, 
                    'model_metrics_mean': model_metrics.mean(), 
                    'model_metrics_std': model_metrics.std(),
                    'trained_model': model_trained
                 }
            )

        return output


    def _fit_sklearn_models(self, model):
        """
        Trains scikit-learn models using spatial cross-validation.

        Args:
            model: Scikit-learn model to be trained.

        Returns:
            tuple: RMSE metrics and trained model.
        """

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        # TODO: ajustar esta Validação cruzada para spatial cross-validation
        # kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        group_kfold = GroupKFold(n_splits=self.k)

        scores = cross_val_score(pipeline, 
                                 X=self.P, y=self.Y,
                                 scoring='neg_mean_squared_error', 
                                 cv=group_kfold.split(self.P, self.Y, groups=self.spatial_groups))

        mse_scores = -scores 
        rmse_scores = np.sqrt(mse_scores)

        model.fit(self.P.values, self.Y.values)

        return rmse_scores, model


    def _fit_ordinary_kriging_models(self, model):
        """
        Trains ordinary kriging models using spatial cross-validation.

        Args:
            model: Ordinary kriging model to be trained.

        Returns:
            tuple: RMSE metrics and trained model.
        """

        rmse_scores = list()
        group_kfold = GroupKFold(n_splits=self.k)

        # TODO: ajustar esta Validação cruzada para spatial cross-validation
        # kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

        # kf_idxs = [x for x in kf.split(self.Y.index)]

        for train_idx, test_idx in group_kfold.split(self.X, self.Y, groups=self.spatial_groups):
        # for k in range(self.k):

            X_train, X_test = self.X.iloc[train_idx].copy(), self.X.iloc[test_idx].copy()
            Y_train, Y_test = self.Y.iloc[train_idx].copy(), self.Y.iloc[test_idx].copy()

            # X_train = self.X.loc[kf_idxs[k][0]].copy()
            # X_test = self.X.loc[kf_idxs[k][1]].copy()

            # Y_train = self.Y.loc[kf_idxs[k][0]].copy()
            # Y_test = self.Y.loc[kf_idxs[k][1]].copy()


            X_train = X_train.values
            X_test = X_test.values

            Y_train = Y_train.values
            Y_test = Y_test.values


            model.fit(
                X = X_train,
                y = Y_train
            )


            predictions = model.predict(
                X = X_test
            )

            predictions = predictions.astype(np.float32)


            mse = mean_squared_error(Y_test, predictions)

            rmse = np.sqrt(mse)

            rmse_scores.append(rmse)


        model.fit(X = self.X.values, y = self.Y.values)

        return np.array(rmse_scores), model


    def _fit_regression_kriging(self, model):
        """
        Trains regression kriging models using spatial cross-validation.

        Args:
            model: Regression kriging model to be trained.

        Returns:
            tuple: RMSE metrics and trained model.
        """

        rmse_scores = list()
        group_kfold = GroupKFold(n_splits=self.k)


        for train_idx, test_idx in group_kfold.split(self.P, self.Y, groups=self.spatial_groups):

            X_train, X_test = self.X.iloc[train_idx].copy(), self.X.iloc[test_idx].copy()
            P_train, P_test = self.P.iloc[train_idx].copy(), self.P.iloc[test_idx].copy()
            Y_train, Y_test = self.Y.iloc[train_idx].copy(), self.Y.iloc[test_idx].copy()

            X_train = X_train.values
            X_test = X_test.values

            P_train = P_train.values
            P_test = P_test.values

            Y_train = Y_train.values
            Y_test = Y_test.values


            Y_train = np.where(np.isinf(Y_train), np.nan, Y_train)
            Y_test  = np.where(np.isinf(Y_test),  np.nan, Y_test)

            Y_train = np.where(np.isnan(Y_train), np.nanmean(Y_train), Y_train)
            Y_test  = np.where(np.isnan(Y_test),  np.nanmean(Y_test),  Y_test)


            model.fit(
                p = P_train,
                x = X_train,
                y = Y_train
            )

            predictions = model.predict(
                p = P_test,
                x = X_test
            )

            predictions = predictions.astype(np.float32)


            mse = mean_squared_error(Y_test, predictions)

            rmse = np.sqrt(mse)

            rmse_scores.append(rmse)


        model.fit(p = self.P.values, x = self.X.values, y = self.Y.values)

        rmse_scores = np.array(rmse_scores)

        return (rmse_scores, model)
