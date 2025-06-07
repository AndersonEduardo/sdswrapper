import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sdswrapper.sampler import SampleGenerator
from sdswrapper.models import Models

from sklearn.metrics import mean_squared_error

from rasterio.mask import mask
from rasterio.plot import show

from pykrige.rk import RegressionKriging
from pykrige.ok import OrdinaryKriging
from sdswrapper.ordinarykriginginterface import OrdinaryKrigingInterface



class Wrapper:
    """
    Wrapper class for managing model training, prediction, and evaluation.

    Attributes:
        model_list (list): List of models to be trained.
        dataset (pd.DataFrame): Dataset containing input and output data.
        X_colum_names (str): Names of input columns.
        p_colum_names (str): Names of projection columns.
        y_column_name (str): Name of the output column.
        projections_folder (str): Path to save projections.
        k (int): Number of folds for cross-validation.
        gridsearch_parameters (list): List of tuples for grid search parameters.
    """

    def __init__(self, 
                 model_list: list,
                 dataset: pd.DataFrame,
                 X_column_names: list,
                 P_column_names: list,
                 y_column_name: str,
                 projections_folder: str,
                 k: int,
                 gridsearch_parameters: list = None):
        """
        Initializes the Wrapper class with the provided parameters.

        Args:
            model_list (list): List of models to be trained.
            dataset (pd.DataFrame): Dataset containing input and output data.
            X_column_names (list): Names of input columns.
            P_column_names (list): Names of projection columns.
            y_column_name (str): Name of the output column.
            projections_folder (str): Path to save projections.
            k (int): Number of folds for cross-validation.
            gridsearch_parameters (list, optional): List of tuples for grid search parameters.
        """
        self.model_list = self.set_model_list(model_list)
        self.dataset = self.set_dataset(dataset)
        self.X_colum_names = self.set_X_column_names(X_column_names)
        self.p_colum_names = self.set_p_column_names(P_column_names)
        self.y_column_name = self.set_y_column_name(y_column_name)
        self.gridsearch_parameters = self.set_gridsearch_parameters(gridsearch_parameters)
        # self.projector = projector # TODO
        self.k = self.set_k(k)
        # self.tag = tag
        self.projections_folder = self.set_projections_folder(projections_folder)


    def set_model_list(self, model_list: list):
        """
        Sets the model list for training.

        Args:
            model_list (list): List of models to be trained.

        Returns:
            list: Validated model list.
        """

        if not isinstance(model_list, list):

            raise TypeError('`model_list` must be a python list.')

        return model_list


    def set_dataset(self, dataset: pd.DataFrame):
        """
        Sets the dataset for training.
        Args:
            dataset (pd.DataFrame): Dataset containing input and output data.
        Returns:
            pd.DataFrame: Validated dataset.
        """

        if not isinstance(dataset, pd.DataFrame):

            raise TypeError('`dataset` must be a pandas DataFrame.')

        return dataset


    def set_X_column_names(self, X_column_names: list):
        """
        Sets the names of input columns.
        Args:
            X_column_names (list): Names of input columns.
        Returns:
            list: Validated names of input columns.
        """

        if not isinstance(X_column_names, list):

            raise TypeError('`X_column_names` must be a python list.')

        if len(X_column_names) == 0:

            raise ValueError('`X_column_names` cannot be an empty list.')

        return X_column_names


    def set_p_column_names(self, P_column_names: list):
        """
        Sets the names of projection columns.
        Args:
            P_column_names (list): Names of projection columns.
        Returns:
            list: Validated names of projection columns.
        """

        if not isinstance(P_column_names, list):

            raise TypeError('`P_column_names` must be a python list.')

        if len(P_column_names) == 0:

            return None

        return P_column_names


    def set_y_column_name(self, y_column_name: str):
        """
        Sets the name of the output column.
        Args:
            y_column_name (str): Name of the output column.
        Returns:
            str: Validated name of the output column.
        """

        if not isinstance(y_column_name, str):

            raise TypeError('`y_column_name` must be a string.')

        if len(y_column_name) == 0:

            raise ValueError('`y_column_name` cannot be an empty string.')

        return y_column_name
    

    def set_gridsearch_parameters(self, gridsearch_parameters: list):
        """
        Sets the grid search parameters for model training.

        Args:
            gridsearch_parameters (list): List of tuples for grid search parameters.

        Returns:
            list: Validated grid search parameters.
        """

        if gridsearch_parameters is None:

            return []

        if not isinstance(gridsearch_parameters, list):

            raise TypeError('`gridsearch_parameters` must be a python list.')

        if len(gridsearch_parameters) == 0:

            return []

        return gridsearch_parameters


    def set_k(self, k: int):
        """
        Sets the number of folds for cross-validation.

        Args:
            k (int): Number of folds for cross-validation.

        Returns:
            int: Validated number of folds.
        """

        if not isinstance(k, int):

            raise TypeError('`k` must be an integer.')

        if k <= 0:

            raise ValueError('`k` must be a positive integer.')

        return k


    def set_projections_folder(self, projections_folder: str):
        """
        Sets the path to save projections.

        Args:
            projections_folder (str): Path to save projections.

        Returns:
            str: Validated path to save projections.
        """

        if not isinstance(projections_folder, str):

            raise TypeError('`projections_folder` must be a string.')

        if len(projections_folder) == 0:

            raise ValueError('`projections_folder` cannot be an empty string.')
        
        if not os.path.exists(projections_folder):

            raise ValueError(f'`projections_folder` does not exist: {projections_folder}')

        return projections_folder


    def fit(self):
        """
        Trains the models using the provided dataset and parameters.

        Returns:
            dict: Profiles of trained models.
        """

        model_profiles = dict()

        try:

            data = self.__check_data(self.dataset)

        except Exception as e:

            raise Exception(e)


        X = data[self.X_colum_names].astype(float).copy()

        p = data[self.p_colum_names].astype(float).copy() \
            if (self.p_colum_names != None) and (len(self.p_colum_names) > 0) else pd.DataFrame()

        y = data[self.y_column_name].astype(float).copy()


        try:

            models = self.set_models(X, p, y)

            model_profiles = models.fit()

        except Exception as e:

            raise Exception(e)


        return model_profiles


    def set_models(self, X: pd.DataFrame, p: pd.DataFrame, y: pd.Series):
        """
        Configures the models with the provided data.

        Args:
            X (pd.DataFrame): Input data.
            p (pd.DataFrame): Projection data.
            y (pd.Series): Output data.

        Returns:
            Models: Configured models instance.
        """
        return Models(
            models = self.model_list,
            X = X,
            p = p,
            y = y,
            k = self.k,
            projections_folder = self.projections_folder
        )


    def __check_data(self, data: pd.DataFrame, numpy_array: bool = False):
        """
        Validates the dataset for NaN or infinite values.

        Args:
            data (pd.DataFrame): Dataset to validate.
            numpy_array (bool): Whether the data is a numpy array.

        Returns:
            pd.DataFrame: Validated dataset.

        Raises:
            Exception: If the dataset contains NaN or infinite values.
        """
        if numpy_array:

            # NaN
            if np.isnan(data).any().any():

                raise Exception("`data` contains NaN values.")

            # Inf
            if np.isinf(data).any().any():

                raise Exception("`data` contains infinite values.")

            return data

        else:

            # NaN
            if np.isnan(data).values.any().any():

                raise Exception("`data` contains NaN values.")

            # Inf
            if np.isinf(data).values.any().any():

                raise Exception("`data` contains infinite values.")

            return data


    @staticmethod
    def predict(trained_model: Models | list, X: pd.DataFrame, 
                p: pd.DataFrame = None, shape: tuple = None):
        """
        Makes predictions using the trained models.

        Args:
            trained_model (Models | list): Trained model or list of models.
            X (pd.DataFrame): Input data for prediction.
            p (pd.DataFrame, optional): Projection data for prediction.
            shape (tuple): Shape of the output prediction.

        Returns:
            np.ndarray: Predictions.

        Raises:
            TypeError: If input types are incorrect.
        """
        if not isinstance(shape, tuple):

            raise TypeError('`shape` must be a tuple.')

        if not isinstance(X, pd.DataFrame):

            raise TypeError('`X` must be a pandas DataFrame.')

        if not isinstance(p, pd.DataFrame):

            raise TypeError('`p` must be a pandas DataFrame.')


        X = X.astype(float).values.copy()
        p = p.astype(float).values.copy()


        if not isinstance(trained_model, list):

            prediction = Wrapper._predict_single_model(trained_model, X, p, shape)

        elif isinstance(trained_model, list):

            prediction = Wrapper._predict_multiple_models(trained_model, X, p, shape)

        else:

            raise TypeError('`trained_model` must be a RegressionKriging, OrdinaryKrigingInterface, sklearn model instance, or a python list instance.')
        
        return prediction


    @staticmethod
    def _predict_multiple_models(trained_models: list, X: np.ndarray, 
                                 p: np.ndarray = None, shape: tuple = None):
        """
        Makes predictions using multiple trained models.

        Args:
            trained_models (list): List of trained models.
            X (np.ndarray): Input data for prediction.
            p (np.ndarray, optional): Projection data for prediction.
            shape (tuple): Shape of the output prediction.

        Returns:
            np.ndarray: Predictions from multiple models.

        Raises:
            TypeError: If input types are incorrect.
        """
        if not isinstance(shape, tuple):

            raise TypeError('`shape` must be a tuple.')

        if not isinstance(X, np.ndarray):

            raise TypeError('`X` must be a numpy array.')

        if not isinstance(p, np.ndarray):

            raise TypeError('`p` must be a numpy array.')

        if not isinstance(trained_models, list):

            raise TypeError('`trained_models` must be a python list.')

        predictions = []

        for model in trained_models:

            prediction = Wrapper._predict_single_model(model['trained_model'], X, p, shape)

            predictions.append(prediction)

        return np.array(predictions)


    @staticmethod
    def _predict_single_model(model: Models, X: np.ndarray, 
                              p: np.ndarray = None, shape: tuple = None):
        """
        Makes predictions using a single trained model.

        Args:
            model (Models): Trained model.
            X (np.ndarray): Input data for prediction.
            p (np.ndarray, optional): Projection data for prediction.
            shape (tuple): Shape of the output prediction.

        Returns:
            np.ndarray: Predictions from the model.

        Raises:
            TypeError: If input types are incorrect.
        """
        if not isinstance(shape, tuple):

            raise TypeError('`shape` must be a tuple.')

        if not isinstance(X, np.ndarray):

            raise TypeError('`X` must be a numpy array.')

        if not isinstance(p, np.ndarray):

            raise TypeError('`p` must be a numpy array.')


        if isinstance(model, RegressionKriging):

            prediction = model.predict(X, p)

        elif isinstance(model, OrdinaryKrigingInterface):

            prediction = model.predict(X)

        else:

            try:

                prediction = model.predict(p)

            except:

                raise TypeError('`model` must be a RegressionKriging, OrdinaryKrigingInterface or a sklearn model instance.')

        return prediction.reshape(shape)


    @staticmethod
    def mask(prediction: np.ndarray, nan_mask: np.ndarray):
        """
        Applies a mask to the prediction to handle NaN values.

        Args:
            prediction (np.ndarray): Prediction data.
            nan_mask (np.ndarray): Mask to apply.

        Returns:
            np.ndarray: Masked prediction.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If shapes of prediction and mask do not match.
        """
        if not isinstance(prediction, np.ndarray):

            raise TypeError('`prediction` must be a numpy array.')

        if not isinstance(nan_mask, np.ndarray):

            raise TypeError('`nan_mask` must be a numpy array.')
        
        if prediction.shape != nan_mask.shape:

            raise ValueError('`prediction` and `nan_mask` must have the same shape.')


        masked_prediction = np.where(np.isnan(nan_mask), np.nan, prediction)


        return masked_prediction


    @staticmethod
    def score(prediction: np.ndarray, reference_data: np.ndarray):
        """
        Calculates the RMSE score between prediction and reference data.

        Args:
            prediction (np.ndarray): Predicted data.
            reference_data (np.ndarray): Reference data.

        Returns:
            float: RMSE score.

        Raises:
            TypeError: If input types are incorrect.
        """
        if not isinstance(prediction, np.ndarray):

            raise TypeError('`prediction` must be a numpy array.')

        if not isinstance(reference_data, np.ndarray):

            raise TypeError('`reference_data` must be a numpy array.')

        mse = mean_squared_error(
            prediction.flatten(),
            reference_data.flatten()
        )

        return float(np.sqrt(mse))


    @staticmethod
    def plot(predictions: pd.DataFrame, save: bool = False, folder: str = None, best: bool = True, **kwargs):
        """
        Plots the predictions and optionally saves the plots.

        Args:
            predictions (pd.DataFrame): DataFrame containing predictions.
            save (bool): Whether to save the plots.
            folder (str, optional): Folder to save the plots.
            best (bool): Whether to plot only the best prediction.
            **kwargs: Additional arguments for customization.

        Raises:
            TypeError: If input types are incorrect.
        """
        column_name = kwargs.get('column_name', None)

        if (column_name is not None) and (not isinstance(column_name, str)):

            raise TypeError('`column_name` must be a string.')

        if not isinstance(predictions, pd.DataFrame):

            raise TypeError('`predictions` must be a pandas DataFrame.')

        if (folder is not None) and  (not isinstance(save, bool)):

            raise TypeError('`save` must be a boolean.')

        if (folder is not None) and (not isinstance(folder, str)):

            raise TypeError('`folder` must be a string.')


        reference_data = kwargs.get('reference_data', None)

        n_cols = 1 if reference_data is None else 3


        if best:

            column_name = 'model_metrics_mean' if column_name is None else column_name

            predictions = predictions.sort_values(by=column_name, ascending=True).iloc[[0]].copy()

            predictions.reset_index(drop=True, inplace=True)

    
        if reference_data is None:

            for i in range(predictions.shape[0]):
                
                prediction = predictions.iloc[i]['prediction']
                name = predictions.iloc[i]['name']

                fig = plt.figure(figsize=(12, 5))

                plt.imshow(prediction, cmap='coolwarm', vmin=0, vmax=1000)
                plt.colorbar()
                plt.title(name) #, fontsize=16)

                # Adicionando pontos do dataset (descomente se desejar)
                # ax1.scatter(
                #     experiment_pipeline.df_full_data['coordenada_X'],
                #     experiment_pipeline.df_full_data['coordenada_Y'],
                #     c='red', s=10, label='Dataset Points'
                # )
                # ax1.legend()

                # plt.tight_layout()

                if save:

                    if not os.path.exists(folder):

                        os.makedirs(folder)

                    plt.savefig(os.path.join(folder, f'{i}_{name}.png'))

                    plt.close()

                else:

                    plt.show()

        else:

            for i in range(predictions.shape[0]):
                
                prediction = predictions.iloc[i]['prediction']
                name = predictions.iloc[i]['name']

                fig = plt.figure(figsize=(12, 5))

                # Mapa observado
                ax1 = plt.subplot(1, n_cols, 1)
                im1 = ax1.imshow(reference_data, cmap='coolwarm', vmin=0, vmax=1000)
                # plt.colorbar(im1, ax=ax1)
                ax1.set_title('Observed') #, fontsize=16)
                
                # Adicionando pontos do dataset (descomente se desejar)
                # ax1.scatter(
                #     experiment_pipeline.df_full_data['coordenada_X'],
                #     experiment_pipeline.df_full_data['coordenada_Y'],
                #     c='red', s=10, label='Dataset Points'
                # )
                # ax1.legend()

                # Mapa previsto
                ax2 = plt.subplot(1, n_cols, 2)
                im2 = ax2.imshow(prediction, origin='lower', cmap='coolwarm', vmin=0, vmax=1000)
                # plt.colorbar(im2, ax=ax2)
                ax2.invert_yaxis()
                ax2.set_title(name) #, fontsize=16)


                # Mapa de diferença
                ax3 = plt.subplot(1, n_cols, 3)
                diff = prediction - reference_data
                im3 = ax3.imshow(diff, origin='lower', cmap='coolwarm', vmin=-1000, vmax=1000)
                plt.colorbar(im3, ax=ax3)
                ax3.invert_yaxis()
                ax3.set_title('Difference') #, fontsize=16)

                plt.tight_layout()
                # plt.show()

                if save:

                    if not os.path.exists(folder):

                        os.makedirs(folder)

                    plt.savefig(os.path.join(folder, f'{i}_{name}.png'))

                    plt.close()

                else:

                    plt.show()
