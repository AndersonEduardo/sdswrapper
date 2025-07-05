import os
import copy
import random
import rasterio
import numpy as np
import pandas as pd
import rasterio.mask

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

    def __init__(self, sample_file:str = None, features:str = None,
                 probability_surface:str = None,
                 reference_polygon:list = None) -> None:
        """
        Initializes the SampleGenerator class with the provided file paths.

        Args:
            y_filepath (str): Path to the y file.
            p_1 (str): Path to the bioclimatic file 01.
            p_2 (str): Path to the bioclimatic file 12.
            georreferenced_raster_filepath (str): Path to the georreferenced raster file.
        """
        self.reference_polygon = \
            self.set_reference_polygon(reference_polygon)
        self.sample_file = self.set_sample_file(sample_file)
        self.features = self.set_features(features)
        self.probability_surface = self.set_probability_surface(probability_surface)



    def set_sample_file(self, sample_file: str):
        """
        Loads the sample file.

        Args:
            sample_file (str): Path to the sample file.

        Returns:
            pd.DataFrame: Loaded sample data.
        """

        if sample_file is None:

            return None


        if isinstance(sample_file, str):

            if os.path.exists(sample_file) == False:

                raise FileNotFoundError(f"Sample file not found: {sample_file}")

            df_sample_file = pd.read_excel(sample_file)

            lat_lon_check = [('lat' in col_name.lower()) or ('lon' in col_name.lower()) 
                             for col_name in df_sample_file.columns]
            
            if sum(lat_lon_check) < 2:

                raise ValueError('It is mandatory the columns "lat" and "lon" on the sample file.')
            
            return df_sample_file

        # TODO: verificar se implementar para input raster (instancia do rasterio)


        else:

            raise TypeError("sample_file must be a string representing the file path to an Excel file.")


    def set_features(self, features:str):
        """
        Loads the features data.
        Args:
            features (str): Path to the features file.
        Returns:
            pd.DataFrame: Loaded features data.
        """

        if features is None:

            return None


        if isinstance(features, str):

            if os.path.isdir(features) == False:

                raise FileNotFoundError(f"Features folder not found: {features}")


            features_list = list()

            for filepath in os.listdir(features):

                if filepath.endswith('.tif') or filepath.endswith('.asc'):

                    feature_name = os.path.splitext(filepath)[0]

                    with rasterio.open(os.path.join(features, filepath)) as raster:

                        if self.reference_polygon is None:

                            self.reference_polygon = self.get_polygon(raster)

                        raster_data = rasterio.mask.mask(raster, self.reference_polygon, crop=True)[0][0]
                        raster_data = copy.deepcopy(raster_data)
                        # raster = raster.read(1)

                        if raster_data.ndim != 2:

                            raise ValueError("Probability surface data must be a 2D array.")

                        features_list.append({'name': feature_name, 'raster': raster_data})

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

        if probability_surface is None:

            return None


        if isinstance(probability_surface, str):

            if os.path.exists(probability_surface) == False:

                raise FileNotFoundError(f"Probability surface file not found: {probability_surface}")

            with rasterio.open(probability_surface) as raster:

                if self.reference_polygon is None:

                    self.reference_polygon = self.get_polygon(raster)

                raster_data = rasterio.mask.mask(raster, self.reference_polygon, crop=True)[0][0]
                # raster_data = raster.read(1)

                if raster_data.ndim != 2:

                    raise ValueError("Probability surface data must be a 2D array.")
                
                if raster_data.min() < 0:

                    raise ValueError("Probability surface data must not contain negative values.")
    
                if raster_data.max() > 1:

                    raise ValueError("Probability surface data must not contain values greater than 1.")
                
                if np.isnan(raster_data).any():

                    raise ValueError("Probability surface data must not contain NaN values.")

                return raster_data


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

        if reference_polygon is None:

            return None
        
        if not isinstance(reference_polygon, list):
            
            raise TypeError("reference_polygon must be a list of coordinates defining the polygon.")


        return reference_polygon


    @staticmethod
    def get_polygon(georreferenced_raster):
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


    @staticmethod
    def get_masked_data(raster_data: np.ndarray|list, polygon: list):
        """
        Applies a mask to the raster data based on the provided polygon.

        Args:
            raster_data (np.array): Raster data to be masked.
            polygon (list): Polygon to apply the mask.

        Returns:
            np.array: Masked raster data.
        """

        if (not isinstance(raster_data, np.ndarray)) and (not isinstance(raster_data, list)):

            raise TypeError("raster_data must be a 2D numpy array or a 1D list with arrays.")
        
        if not isinstance(polygon, list):

            raise TypeError("polygon must be a list of coordinates defining the polygon.")


        if isinstance(raster_data, np.ndarray):

            data_masked, transform = rasterio.mask.mask(raster_data, polygon, crop=True)

            return data_masked[0]
        
        elif isinstance(raster_data, list):

            masked_data = []

            for data in raster_data:

                if isinstance(data, np.ndarray):

                    data_masked, transform = rasterio.mask.mask(data, polygon, crop=True)

                    masked_data.append(data_masked[0])
                
                else:

                    raise TypeError("Each item in raster_data list must be a numpy array.")


            return masked_data
        
        else:

            raise TypeError("raster_data must be a 2D numpy array or a list of numpy arrays.")


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

            raise Exception("Sample_size is larger than the number of valid coordinates. Returning all valid coordinates.")


        if use_probs:

            sampled_coordinates = np.random.choice(
                len(valid_coordinates), 
                size = sample_size, 
                replace = False, 
                p = valid_probabilities
            )


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

        if self.probability_surface is None:

            raise ValueError("Probability surface is not set. Please provide a valid probability surface.")


        if pseudoabsences == True:

            # valores de 0 a 1% = áreas de pseudo-absences
            data_processed = np.where(
                (self.probability_surface > 0.) & (self.probability_surface < 0.1), 
                1.,
                0.
            )

        elif pseudoabsences == False:

            data_processed = np.where(
                self.probability_surface > 0., 
                self.probability_surface, 
                0.
            )    


        else:

            raise ValueError("`pseudoabsences` must be a python bool.")


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

        if self.probability_surface is None:

            raise ValueError("Probability surface is not set. Please provide a valid probability surface.")


        sampled_coords = self.get_sample_coordinates(n, pseudoabsences=pseudoabsences)
        features = [] if self.features is None else self.features

        sample_output = list()
        sample_row = dict()


        for coord in sampled_coords:

            sample_row.update(
                {
                    'lat': coord[1], 
                    'lon': coord[0]
                    }
            )

            for feature in features:

                sample_row.update(
                    {
                        feature['name']: feature['raster'][coord[1], coord[0]]
                    }
                )

            sample_output.append(sample_row.copy())


        return pd.DataFrame(sample_output).astype(np.float64)



    def get_full_data(self):
        """
        Combines all available data into a single DataFrame.

        Returns:
            pd.DataFrame: Combined data in DataFrame format.
        """

        if self.features is None:

            raise ValueError("Features file path is not set. Please provide a valid sample file.")


        df_fulldata = pd.DataFrame()


        # nome = None


        for feature in self.features:

            feature_df = array_to_dataframe(feature['raster'])

            feature_df.rename(columns={'value': feature['name']}, inplace=True)


            if df_fulldata.empty:

                df_fulldata = feature_df

            else:

                # df_fulldata = pd.merge(df_fulldata, feature_df, on=['x', 'y'], how='outer')
                df_fulldata = pd.merge(df_fulldata, feature_df, left_index=True, right_index=True, how='outer')


        df_fulldata = df_fulldata.drop(columns=['x_y', 'y_y'])

        df_fulldata.rename(columns={'x_x': 'lon', 'y_x': 'lat'}, inplace=True)


        return df_fulldata
