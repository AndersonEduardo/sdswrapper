import os
import random
import rasterio
import numpy as np
import pandas as pd

from sdswrapper.utils.utils import array_to_dataframe



class SampleGenerator:
    """
    Class for generating samples of features and target data.

    Attributes:
        y_filepath (str): Path to the target data file.
        p_1 (str): Path to the feature 01 datafile.
        p_2 (str): Path to the feature 02 data file.
        georreferenced_raster: Loaded suitability data.
        y: Loaded and adjusted y data.
        bioclim_01: Processed bioclimatic data 01.
        bioclim_12: Processed bioclimatic data 12.
    """

    def __init__(self, sample_file:str, features:str, probability_surface:str, 
                 reference_polygon:list = None) -> None:
        """
        Initializes the SampleGenerator class with the provided file paths.

        Args:
            y_filepath (str): Path to the y file.
            p_1 (str): Path to the bioclimatic file 01.
            p_2 (str): Path to the bioclimatic file 12.
            georreferenced_raster_filepath (str): Path to the georreferenced raster file.
        """

        self.sample_file = self.set_sample_file(sample_file)
        self.features = self.set_features(features)
        self.probability_surface = self.set_probability_surface(probability_surface)
        self.reference_polygon = self.set_reference_polygon(reference_polygon)


    def set_sample_file(self, sample_file: str):
        """
        Loads the sample file.

        Args:
            sample_file (str): Path to the sample file.

        Returns:
            pd.DataFrame: Loaded sample data.
        """

        if isinstance(sample_file, str):

            if os.path.exists(sample_file) == False:

                raise FileNotFoundError(f"Sample file not found: {sample_file}")

            return pd.read_excel(sample_file)


        # TODO: implementar para input raster (instancia do rasterio)


        else:

            raise TypeError("sample_file must be a string representing the file path to an Excel file.")


    def set_features(self, features: str):
        """
        Loads the features data.
        Args:
            features (str): Path to the features file.
        Returns:
            pd.DataFrame: Loaded features data.
        """

        if isinstance(features, str):

            if os.path.exists(features) == False:

                raise FileNotFoundError(f"Features file not found: {features}")

            features_list = list()

            for filepath in os.listdir(features):

                if filepath.endswith('.tif') or filepath.endswith('.asc'):

                    raster = rasterio.open(os.path.join(features, filepath))

                    feature_name = os.path.splitext(filepath)[0]

                    features_list.append({'name': feature_name, 'raster': raster})

            return features_list


        # TODO: implementar para input raster (instancia do rasterio)


        else:

            raise TypeError("features must be a string representing the directory path containing raster files.")


    def set_probability_surface(self, probability_surface:str):
        """
        Loads the probability surface data.

        Args:
            probability_surface (str): Path to the probability surface file.

        Returns:
            pd.DataFrame: Loaded probability surface data.
        """

        if isinstance(probability_surface, str):

            if os.path.exists(probability_surface) == False:

                raise FileNotFoundError(f"Probability surface file not found: {probability_surface}")

            return rasterio.open(probability_surface)
        

        # TODO: implementar para input raster (instancia do rasterio)


        else:

            raise TypeError("probability_surface must be a string representing the file path to a raster file.")


    def set_reference_polygon(self, reference_polygon:list):
        """
        Sets the reference polygon for masking.

        Args:
            reference_polygon (list): List of coordinates defining the polygon.

        Returns:
            list: Reference polygon.
        """

        if isinstance(reference_polygon, list):

            return reference_polygon

        else:

            raise TypeError("reference_polygon must be a list of coordinates defining the polygon.")


    def set_georreferenced_raster(self, georreferenced_raster_filepath:str):

        return rasterio.open(georreferenced_raster_filepath)


    def set_y(self, y_filepath: str):
        """
        Loads and adjusts the y data from the provided file.

        Args:
            y_filepath (str): Path to the y file.

        Returns:
            np.ndarray: Adjusted y data.
        """
        y = pd.read_pickle(y_filepath)

        return np.where(y > 1000, 1000, y) # TODO: ajustar aqui (esta assim por conta do caso de  estudo)


    def set_bioclim_01(self, p_1: str):
        """
        Loads and processes the bioclimatic data 01.

        Args:
            p_1 (str): Path to the bioclimatic file 01.

        Returns:
            np.ndarray: Processed bioclimatic data 01.
        """
        p_1 = rasterio.open(p_1)

        p_1 = self.get_masked_data(p_1, self.get_polygon(self.georreferenced_raster))

        return p_1[:-1, :-1]


    def set_bioclim_12(self, p_2: str):
        """
        Loads and processes the bioclimatic data 12.

        Args:
            p_2 (str): Path to the bioclimatic file 12.

        Returns:
            np.ndarray: Processed bioclimatic data 12.
        """
        p_2 = rasterio.open(p_2)

        p_2 = self.get_masked_data(p_2, self.get_polygon(self.georreferenced_raster))

        return p_2[:-1, :-1]


    def get_polygon(self, georreferenced_raster):
        """
        Generates a polygon based on the bounds of the suitability data.

        Args:
            georreferenced_raster: Suitability data.

        Returns:
            list: Polygon representing the data bounds.
        """
        bbox = georreferenced_raster.bounds

        return [{
            "type": "Polygon",
            "coordinates": [[
                (bbox.left, bbox.bottom),
                (bbox.left, bbox.top),
                (bbox.right, bbox.top),
                (bbox.right, bbox.bottom),
                (bbox.left, bbox.bottom)
            ]]
        }]


    def get_masked_data(self, raster_data: np.array, polygon: list):
        """
        Applies a mask to the raster data based on the provided polygon.

        Args:
            raster_data (np.array): Raster data to be masked.
            polygon (list): Polygon to apply the mask.

        Returns:
            np.array: Masked raster data.
        """
        data_masked, transform = rasterio.mask.mask(raster_data, polygon, crop=True)

        return data_masked[0]


    def sample_coordinates(self, data, sample_size, use_probs: bool = False):
        """
        Samples a specified number of coordinates from a 2D array, excluding NaN values.

        Args:
            data (np.array): 2D array from which coordinates will be sampled.
            sample_size (int): Number of coordinates to sample.
            use_probs (bool): Determines if probabilities will be used for sampling.

        Returns:
            list: List of tuples representing coordinates (row, column).
        """

        if not isinstance(use_probs, bool):

            raise Exception("use_probs must be a boolean value")


        valid_coordinates = []
        valid_probabilities = []

        sum_probabilities = np.nansum(data)

        for i in range(data.shape[0]):

            for j in range(data.shape[1]):

                if not np.isnan(data[i, j]):

                    valid_coordinates.append((j, i))
                    valid_probabilities.append( data[i, j]/sum_probabilities )

        # if sum(valid_probabilities) < 1:

        #     offset = 1.0 - sum(valid_probabilities)

        #     for i in range(len(valid_probabilities)):

        #         valid_probabilities[i] += offset/len(valid_probabilities)

        # normalizando
        valid_probabilities = np.array(valid_probabilities)
        valid_probabilities /= valid_probabilities.sum()


        if sample_size > len(valid_coordinates):

            raise Exception("Warning: sample_size is larger than the number of valid coordinates. Returning all valid coordinates.")


        if use_probs:

            sampled_coordinates = np.random.choice(len(valid_coordinates), size=sample_size, replace=False, p=valid_probabilities)

            return np.array(valid_coordinates)[sampled_coordinates]

        else:

            return random.sample(valid_coordinates, sample_size)


    def get_sample_coordinates(self, n: int, pseudoabsences: bool = False):
        """
        Obtains sampled coordinates based on y data.

        Args:
            n (int): Number of coordinates to sample.
            pseudoabsences (bool): Determines if pseudo-absences will be considered.

        Returns:
            list: Sampled coordinates.
        """

        if pseudoabsences == True:

            data_processed = np.where(self.y == 0, 1, 0)

        elif pseudoabsences == False:

            data_processed = np.where(self.y > 0, self.y, 0)

        else:

            raise ValueError("`pseudoabsences` must be a python bool.ß")

        return self.sample_coordinates(
            data = data_processed,
            sample_size = n,
            use_probs = True
        )


    def extract(self, coods: list, raster: np.array):
        """
        Extracts values from a raster based on provided coordinates.

        Args:
            coods (list): List of coordinates (x, y).
            raster (np.array): Raster data from which values will be extracted.

        Returns:
            list: Extracted values from the raster.
        """

        output_values = list()

        for coord in coods:

            output_values.append((coord[0], coord[1], raster[coord[1], coord[0]]))

        return output_values


    def sample(self, n: int, pseudoabsences: bool = False):
        """
        Generates a data sample combining coordinates, y, and features.

        Args:
            n (int): Number of samples to generate.
            pseudoabsences (bool): Determines if pseudo-absences will be considered.

        Returns:
            pd.DataFrame: Sampled data in DataFrame format.
        """

        sampled_coords = self.get_sample_coordinates(n, pseudoabsences=pseudoabsences)

        y_extracted = self.extract(sampled_coords, self.y)

        p_1_extracted = self.extract(sampled_coords, self.p_1)

        p_2_extracted = self.extract(sampled_coords, self.p_2)


        df_values = list()

        for i in range(len(sampled_coords)):

            df_values.append({
                'ID': i,
                'coordenada_X': sampled_coords[i][0],
                'coordenada_Y': sampled_coords[i][1],
                'y': y_extracted[i][-1],
                'p_1': p_1_extracted[i][-1],
                'p_2': p_2_extracted[i][-1]
            })

        return pd.DataFrame(df_values).astype(np.float64)


    def get_full_data(self):
        """
        Combines all available data into a single DataFrame.

        Returns:
            pd.DataFrame: Combined data in DataFrame format.
        """

        y_adjusted = np.where(np.isnan(self.y), 0, self.y)
        p_1_adjusted = np.where(np.isnan(self.p_1), 0, self.p_1)
        p_2_adjusted = np.where(np.isnan(self.p_2), 0, self.p_2)

        df_full_y = array_to_dataframe(y_adjusted)

        df_full_p_1 = array_to_dataframe(p_1_adjusted)

        df_full_p_2 = array_to_dataframe(p_2_adjusted)

        df_fulldata = df_full_y.copy()

        df_fulldata['p_1'] = df_full_p_1['value'].values.astype(np.float64)

        df_fulldata['p_2'] = df_full_p_2['value'].values.astype(np.float64)

        df_fulldata.rename(
            columns = {'x':'coordenada_X', 'y':'coordenada_Y', 'value': 'y'},
            inplace = True
        )

        return df_fulldata