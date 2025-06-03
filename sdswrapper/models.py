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
        Y (pd.DataFrame): Output (target) data.
        k (int): Number of folds for cross-validation.
        projections_folder (str): Path to save projections.
        spatial_groups (np.ndarray): Spatial groups for cross-validation.
    """

    def __init__(self, models: list, X: pd.DataFrame, p: pd.DataFrame, y: pd.DataFrame, 
                 k: int, projections_folder: str) -> None:
        """
        Initializes the Models class with the provided data and settings.

        Args:
            models (list): List of models to be trained.
            X (pd.DataFrame): Coordinate data.
            p (pd.DataFrame): Feature data.
            y (pd.DataFrame): Target data.
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



    def __check_data(self, data: pd.DataFrame):
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

        X = X.reset_index(drop=True).astype('float32')

        X = self.__check_data(X)

        return X


    def set_P(self, p: pd.DataFrame):
        """
        Sets the projection data P after verification and conversion.

        Args:
            p (pd.DataFrame): Projection data.

        Returns:
            pd.DataFrame: Processed projection data.
        """

        if (p.empty) or (p == None):

            return None


        p = p.reset_index(drop=True).astype('float32')

        P = self.__check_data(p)

        return p


    def set_Y(self, y: pd.DataFrame):
        """
        Sets the output data Y after verification and conversion.

        Args:
            y (pd.DataFrame): Output data.

        Returns:
            pd.DataFrame: Processed output data.
        """

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

        if p is None:

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

        print('~~'*20)
        print('self.P:\n', self.P)
        print('~~'*20)


        for name, model in self.models:

            model_type = None

            if isinstance(model, RegressionKriging):

                if self.P == None:

                    print("`P` must be provided for Regression Kriging models. Skipping model:", name)

                    continue

                model_type = 'KR'

                model_metrics, model_trained = self._fit_regression_kriging(model)

            elif isinstance(model, OrdinaryKrigingInterface):

                model_type = 'KR'

                model_metrics, model_trained = self._fit_ordinary_kriging_models(model)

            else:

                if self.P == None:

                    print("`P` must be provided for Regression Kriging models. Skipping model:", name)

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

        # TODO: ajustar esta Validação cruzada para spatial cross-validation
        # kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

        # kf_idxs = [x for x in kf.split(self.Y.index)]

        for train_idx, test_idx in group_kfold.split(self.P, self.Y, groups=self.spatial_groups):
        # for k in range(self.k):

            # X_train = self.X.loc[kf_idxs[k][0]].copy()
            # X_test = self.X.loc[kf_idxs[k][1]].copy()

            # P_train = self.P.loc[kf_idxs[k][0]].copy()
            # P_test = self.P.loc[kf_idxs[k][1]].copy()

            # Y_train = self.Y.loc[kf_idxs[k][0]].copy()
            # Y_test = self.Y.loc[kf_idxs[k][1]].copy()

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


    # def projection(self, X_full:pd.DataFrame, P_full:pd.DataFrame,
    #                shape:tuple, tag:str = None):


    #     if not isinstance(X_full, pd.DataFrame):

    #         raise Exception("`X_full` must be pandas DataFrame.")

    #     if not isinstance(P_full, pd.DataFrame):

    #         raise Exception("`P_full` must be pandas DataFrame.")

    #     # if not isinstance(Y_full, pd.Series):

    #     #     raise Exception("`Y_full` must be pandas DataFrame.")

    #     if not isinstance(shape, tuple):

    #         raise Exception("`shape` must be a tuple.")

    #     if (tag != None) and (not isinstance(tag, str)):

    #         raise Exception("`shape` must be a tuple.")

    #     if np.isinf(X_full).any().any():

    #         raise Exception("`X_full` contains infinite values.")

    #     # if np.isinf(Y_full).any():

    #     #     raise Exception("`Y_full` contains infinite values.")

    #     if np.isinf(P_full).any().any():

    #         raise Exception("`P_full` contains infinite values.")

    #     if np.isnan(X_full).any().any():

    #         raise Exception("`X_full` contains infinite values.")

    #     # if np.isnan(Y_full).any():

    #     #     raise Exception("`Y_full` contains infinite values.")

    #     if np.isnan(P_full).any().any():

    #         raise Exception("`P_full` contains infinite values.")


    #     X_full_raw = X_full.copy()
    #     P_full_raw = P_full.copy()
    #     # Y_full_raw = Y_full.copy()


    #     ################# TODO: apagar depois
    #     # print()
    #     # print('~~'*20)
    #     # print('# PARTE B')
    #     # print('type(X_full):', type(X_full))
    #     # print('type(P_full):', type(P_full))
    #     # print('type(Y_full):', type(Y_full))
    #     # print()
    #     # print('~~'*20)
    #     # print()
    #     #################


    #     if not os.path.exists(self.projections_folder):

    #         os.mkdir(self.projections_folder)


    #     for name, model in self.models:

    #         print('# Model:', name) # TODO: comentar esta linha depois

    #         model_type = None


    #         # Regression Kriging
    #         if isinstance(model, RegressionKriging):

    #             model_type = 'KR'

    #             # X = df_fulldata[X_column_names]
    #             # p = df_fulldata[P_colum_names]
    #             # y = df_fulldata[Y_column_name]

    #             ################# TODO: apagar depois
    #             # print()
    #             # print('~~'*20)
    #             # print(f'# PARTE C em cima - {name}')
    #             # print('type(X_full):', type(X_full))
    #             # print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             # print()
    #             # print('~~'*20)
    #             # print()
    #             #################

    #             y_data = self.Y.values.copy()
    #             y_data = np.where(np.isinf(y_data), np.nan, y_data)
    #             y_data = np.where(np.isnan(y_data), np.nanmean(y_data), y_data)

    #             model.fit(
    #                 x = self.X.values.copy(),
    #                 p = self.P.values.copy(),
    #                 y = y_data
    #             )


    #             X_full = X_full_raw.copy()
    #             X_full = X_full.reset_index(drop=True).copy()
    #             X_full = X_full.values
    #             X_full = np.where(np.isinf(X_full), np.nan, X_full)
    #             X_full = np.where(np.isnan(X_full), np.nanmean(X_full), X_full)

    #             P_full = P_full_raw.copy()
    #             P_full = P_full.reset_index(drop=True).copy()
    #             P_full = P_full.values
    #             P_full = np.where(np.isinf(P_full), np.nan, P_full)
    #             P_full = np.where(np.isnan(P_full), np.nanmean(P_full), P_full)


    #             ################# TODO: apagar depois
    #             # print()
    #             # print('~~'*20)
    #             # print(f'# PARTE C embaixo - {name}')
    #             # print('type(X_full):', type(X_full))
    #             # print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             # print()
    #             # print('~~'*20)
    #             # print()
    #             #################


    #             projection = model.predict(
    #                 x = X_full,
    #                 p = P_full
    #             )

    #             projection = projection.astype(np.float32)
    #             projection = projection.reshape(shape)

    #             projection = np.where(np.isinf(projection), np.nan, projection)
    #             projection = np.where(np.isnan(projection), np.nanmean(projection), projection)

    #             hash = datetime.now().strftime("%Y%m%d_%H%M")
    #             hash = tag + '_' + hash if tag != None else hash

    #             with open(os.path.join(self.projections_folder, f'{name}_projection_{hash}.pkl'), 'wb') as f:

    #                 pickle.dump(projection, f)


    #         # Ordinary Kriging
    #         elif isinstance(model, OrdinaryKrigingInterface):

    #             ################# TODO: apagar depois
    #             # print()
    #             # print('~~'*20)
    #             # print(f'# PARTE D - elif - {name}')
    #             # print('type(X_full):', type(X_full))
    #             # print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             # print()
    #             # print('~~'*20)
    #             # print()
    #             #################

    #             model_type = 'KR'

    #             y_data = self.Y.values.copy()
    #             y_data = np.where(np.isinf(y_data), np.nan, y_data)
    #             y_data = np.where(np.isnan(y_data), np.nanmean(y_data), y_data)

    #             model.fit(
    #                 self.X.values.copy(),
    #                 y_data
    #             )

    #             X_full = X_full_raw.copy()
    #             X_full = X_full.reset_index(drop=True).copy()
    #             X_full = X_full.values
    #             X_full = np.where(np.isinf(X_full), np.nan, X_full)
    #             X_full = np.where(np.isnan(X_full), np.nanmean(X_full), X_full)

    #             projection = model.predict(
    #                 X_full
    #             )

    #             projection = projection.astype(np.float32)
    #             projection = projection.reshape(shape)

    #             projection = np.where(np.isinf(projection), np.nan, projection)
    #             projection = np.where(np.isnan(projection), np.nanmean(projection), projection)

    #             hash = datetime.now().strftime("%Y%m%d_%H%M")
    #             hash = tag + '_' + hash if tag != None else hash

    #             with open(os.path.join(self.projections_folder, f'{name}_projection_{hash}.pkl'), 'wb') as f:

    #                 pickle.dump(projection, f)

    #         else:

    #             ################# TODO: apagar depois
    #             # print()
    #             # print('~~'*20)
    #             # print(f'# PARTE Ea - else - {name}')
    #             # print('type(X_full):', type(X_full))
    #             # print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             # print()
    #             # print('~~'*20)
    #             # print()
    #             #################


    #             model_type = 'SK'

    #             # X = df_fulldata[X_column_names + P_colum_names]
    #             # y = df_fulldata[Y_column_name]

    #             # model.fit(self.XP, self.Y)
    #             model.fit(
    #                 self.P.values.copy(),
    #                 self.Y.values.copy()
    #             )

    #             ################# TODO: apagar depois
    #             # print()
    #             # print('~~'*20)
    #             # print(f'# PARTE Eb - else - {name}')
    #             # print('type(X_full):', type(X_full))
    #             # print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             # print()
    #             # print('~~'*20)
    #             # print()
    #             #################

    #             # X_full.reset_index(drop=True, inplace=True)
    #             # P_full = P_full#.reset_index(drop=True).copy()
    #             # XP = pd.concat([X_full, P_full], axis=1)

    #             ################ TODO: apagar depois
    #             print()
    #             print('~~'*20)
    #             print(f'# PARTE Ec - else - {name}')
    #             print('type(X_full):', type(X_full))
    #             print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             print()
    #             print('~~'*20)
    #             print()
    #             ################

    #             P_full = P_full_raw.copy()
    #             P_full = P_full.astype('float32') #.astype('float32').values.astype(np.float32)

    #             ################ TODO: apagar depois
    #             print()
    #             print('~~'*20)
    #             print(f'# PARTE Ec ALPHA - else - {name}')
    #             print('type(X_full):', type(X_full))
    #             print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             print()
    #             print('~~'*20)
    #             print()
    #             ################

    #             P_full = np.where(np.isinf(P_full), np.nan, P_full)

    #             ################# TODO: apagar depois
    #             print()
    #             print('~~'*20)
    #             print(f'# PARTE Ec BETHA - else - {name}')
    #             print('type(X_full):', type(X_full))
    #             print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             print()
    #             print('~~'*20)
    #             print()
    #             #################

    #             P_full = np.where(np.isnan(P_full), np.nanmean(P_full), P_full)


    #             ################# TODO: apagar depois
    #             print()
    #             print('~~'*20)
    #             print(f'# PARTE Ed - else - {name}')
    #             print('type(X_full):', type(X_full))
    #             print('type(P_full):', type(P_full))
    #             # print('type(Y_full):', type(Y_full))
    #             print()
    #             print('~~'*20)
    #             print()
    #             #################

    #             # projection = model.predict(XP)
    #             projection = model.predict(
    #                 P_full#.values
    #             )

    #             projection = projection.astype(np.float32)
    #             projection = projection.reshape(shape)

    #             projection = np.where(np.isinf(projection), np.nan, projection)
    #             projection = np.where(np.isnan(projection), np.nanmean(projection), projection)

    #             hash = datetime.now().strftime("%Y%m%d_%H%M")
    #             hash = tag + '_' + hash if tag != None else hash

    #             with open(os.path.join(self.projections_folder, f'{name}_projection_{hash}.pkl'), 'wb') as f:

    #                 pickle.dump(projection, f)
