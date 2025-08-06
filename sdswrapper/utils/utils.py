import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: implementar logging corretamente


def load_asc(file_path):

    with open(file_path, 'r') as f:
        header = {}
        for _ in range(6):  # As 6 primeiras linhas são o cabeçalho
            key, value = f.readline().split()
            header[key] = float(value) if '.' in value else int(value)

        # Lendo os dados da matriz
        data = np.loadtxt(f)

    return header, data


def plot_asc(file_path):

    header, data = load_asc(file_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='terrain', origin='upper')
    plt.colorbar(label="Elevação (m)")
    plt.title("Modelo Digital de Elevação (DEM)")
    plt.xlabel("Colunas")
    plt.ylabel("Linhas")
    plt.show()


def replace_with_nan(data):
    """Substitui -9999 por NaN em um array NumPy."""

    data = np.where(data == -9999, np.nan, data)

    return data


def apply_function_to_matrix(data, func):
    """Applies a function to each element of the matrix.

    Args:
        data: The input matrix (NumPy array).
        func: The function to apply to each element.

    Returns:
        A new matrix with the function applied to each element.
    """

    new_data = np.vectorize(func)(data)

    return new_data


def array_to_dataframe(array:np.array):

    output_values = list()

    for y in range(array.shape[0]):

        for x in range(array.shape[1]):

            output_values.append((x, y, array[y, x]))

    return pd.DataFrame(output_values, columns=['x', 'y', 'value'])


def pad_array(array, target_h, target_w, pad_value=np.nan):
    """Faz padding em uma array 2D até o tamanho desejado."""

    h, w = array.shape
    pad_h = target_h - h
    pad_w = target_w - w

    return np.pad(array, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
