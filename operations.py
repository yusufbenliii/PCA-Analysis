import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
In this file numpy library is used for creating numbers and also creating arrays
because when I try to do calculations with python defaults I realized that there are
little differences on numbers, those differences might cause the errors that's why I prefer to use numpy.
Addition to numpy the seaborn and matplotlib libraries were used.
These libraries just for plotting the graph's of the result.
'''


def my_standard_dev(lst):
    length = np.float64(len(lst))
    mean = my_mean(lst)
    deviation = np.float64(0)
    for number in lst:
        deviation += pow((number - mean), 2)
    deviation = pow((deviation / (length - 1)), 0.5)
    return deviation


def my_mean(lst):
    length = np.float64(len(lst))
    mean = np.float64(0)
    for number in lst:
        mean += number
    mean = mean / length
    return mean


def my_covariance(x, y):
    length = len(x)
    x_mean = my_mean(x)
    y_mean = my_mean(y)
    cov = np.float64(0)
    for i in range(length):
        cov += (x[i] - x_mean) * (y[i] - y_mean)
    cov = cov / length
    return cov


def column_matrices(data_set):
    col_matrices = {}
    for column in range(len(data_set[0])):
        key = f'column{column}'
        col_matrices[key] = []  # create empty list for current column with a key
        for row in range(len(data_set)):
            col_matrices[key].append(data_set[row][column].astype('float64'))
    return col_matrices


def column_means(data_set):
    col_matrices = column_matrices(data_set)
    col_means = {}
    for column in range(len(data_set[0])):
        key = f'column{column}'
        column_mean = my_mean(col_matrices[key])
        col_means[key] = column_mean
    return col_means


def column_standard_devs(data_set):
    col_matrices = column_matrices(data_set)
    col_sdevs = {}
    for column in range(len(data_set[0])):
        key = f'column{column}'
        column_standard_dev = my_standard_dev(col_matrices[key])
        col_sdevs[key] = column_standard_dev.astype('float64')
    return col_sdevs


def transpose(matrix):
    row_len = len(matrix[0])
    column_len = len(matrix)
    T = np.arange(row_len * column_len, dtype=np.float64).reshape(row_len, column_len)
    for row in range(column_len):
        for column in range(row_len):
            T[column][row] = matrix[row][column]

    return T


def dot_product(a, b):
    row_len_a = len(a)
    column_len_a = len(a[0])
    column_len_b = len(b[0])
    dot_matrix = np.arange(row_len_a * column_len_b, dtype=np.float64).reshape(row_len_a, column_len_b)
    for row_a in range(row_len_a):
        for col_b in range(column_len_b):
            dot_of_index = 0
            for col_a in range(column_len_a):
                dot_of_index += a[row_a][col_a] * b[col_a][col_b]
            dot_matrix[row_a][col_b] = dot_of_index
    return dot_matrix


'''
fit_data function is same as sklearn library's fit_transform() fucntion
    x′=(x−μ)/σ
'''


def fit_data(data_set):
    data_len_column = len(data_set[0])
    data_len_row = len(data_set)
    fitted_column_matrix = np.arange(data_len_row * data_len_column, dtype=np.float64).reshape(data_len_row,
                                                                                               data_len_column)
    col_matrices = column_matrices(data_set)
    col_means = column_means(data_set)
    col_sdevs = column_standard_devs(data_set)
    for column in range(data_len_column):
        key = f"column{column}"
        current_column_mean = col_means[key]
        current_column_sdev = col_sdevs[key]
        for row in range(data_len_row):
            current_number = col_matrices[key][row]
            fitted_current_number = (current_number - current_column_mean) / current_column_sdev
            fitted_column_matrix[row][column] = fitted_current_number

    return fitted_column_matrix


def covariance_matrix(col_matrices):
    '''
    Here we find all covariance combinations.
    We are calculating same values twice for example:
    cov(column0, column1)  and cov(column1, column0) values are same
    but it will be have benefits to us while creating covariance matrix.
    '''
    cov_values = {}
    for key in col_matrices:
        cov_index_zero = col_matrices[key]
        for i in range(len(col_matrices)):
            current_key = f"column{i}"
            cov_index_next = col_matrices[current_key]
            cov = my_covariance(cov_index_zero, cov_index_next)
            cov_value_key = f"{key}+{current_key}"
            cov_values[cov_value_key] = cov

    size_of_column_matrices = len(col_matrices)
    cov_matrix = np.arange(size_of_column_matrices ** 2, dtype=np.float64).reshape(size_of_column_matrices,
                                                                                   size_of_column_matrices)
    row_len = column_len = len(cov_matrix)
    for row in range(row_len):
        for column in range(column_len):
            key_for_element = f"column{row}+column{column}"
            element_from_key = cov_values[key_for_element]
            cov_matrix[row][column] = element_from_key

    return cov_matrix


def variance(eig_vals):
    '''
    This function is for calculating the variance of the eigen values
    '''
    def sum_of_list(lst):
        sum_of_elements = np.float64(0)
        for i in lst:
            sum_of_elements += i
        return sum_of_elements

    eig_variances = []
    sum_of_values = sum_of_list(eig_vals)
    for i in range(len(eig_vals)):
        eig_variances.append(eig_vals[i] / sum_of_values)

    return eig_variances


def eig_vec_plot(score, eig, original_data):
    '''
    This function just for plotting
    '''
    xs = score[:, 0]
    ys = score[:, 1]
    n = eig.shape[0]
    sns.scatterplot(xs, ys, hue=original_data['Brand'], s=100)
    for i in range(n):
        plt.arrow(0, 0, eig[i][0], eig[i][1], color='black')
        plt.text(eig[i][0] * 1.15, eig[i][1] * 1.15, f"EIGEN{i+1}", color='black', ha='center', va='center')
