import numpy as np
import scipy.stats as sps
from lewin_livnat_zwick import *
from tqdm import tqdm
from time import time

def test_benchmark(n, sample_count=100):
  """
  Тестирует алгоритм Левина-Ливната-Цвика

  :param n: число переменных и дизъюнктов
  :param sample_count: число примеров
  :return: значения разных статистик для аппроксимаций и времен работы
  """
  samples = sps.randint.rvs(1, 2 * n + 1, size=(sample_count, 2 * n), random_state=42)

  var_values = np.array([list(f'{i:0{n}b}') for i in range(2 ** n)], dtype=int)
  combined_table = np.hstack((var_values, 1 - var_values))

  approximations = []
  times = []

  for clauses_list in tqdm(samples):
    indexed_values = combined_table[:, clauses_list - 1]
    satisfied_clauses_counts = np.sum((indexed_values[:, ::2] + indexed_values[:, 1::2]) > 0, axis=1)
    max_satisfied_clauses = np.max(satisfied_clauses_counts)

    start_time = time()
    result = np.array(lewin_livnat_zwick(rotation, n, clauses_list), dtype=int)
    end_time = time()
    times.append(end_time - start_time)

    final_result = np.concatenate([result, 1 - result])
    final_values = final_result[clauses_list - 1]
    final_satisfiend_clauses_count = np.sum((final_values[::2] + final_values[1::2]) > 0)
    approximations.append(final_satisfiend_clauses_count / max_satisfied_clauses)

  approx_q25 = np.percentile(approximations, 25)
  approx_q75 = np.percentile(approximations, 75)
  approx_iqr = approx_q75 - approx_q25
  approx_median = np.median(approximations)

  time_q25 = np.percentile(times, 25)
  time_q75 = np.percentile(times, 75)
  time_iqr = time_q75 - time_q25
  time_median = np.median(times)

  approx_left_outliers = np.sum(approximations < approx_median - 1.5 * approx_iqr)
  approx_right_outliers = np.sum(approximations > approx_median + 1.5 * approx_iqr)
  time_left_outliers = np.sum(times < time_median - 1.5 * time_iqr)
  time_right_outliers = np.sum(times > time_median + 1.5 * time_iqr)

  approx_stats = {
    'min': np.min(approximations),
    'max': np.max(approximations),
    'mean': np.mean(approximations),
    'stddev': np.std(approximations),
    'median': approx_median,
    'iqr': approx_iqr,
    'outliers': f'{approx_left_outliers};{approx_right_outliers}'
  }

  time_stats = {
    'min': np.min(times),
    'max': np.max(times),
    'mean': np.mean(times),
    'stddev': np.std(times),
    'median': time_median,
    'iqr': time_iqr,
    'outliers': f"{time_left_outliers};{time_right_outliers}"
  }

  print(f'\nResults for n={n}:')
  print('Approximation statistics (ratios):')
  print(f"Min: {approx_stats['min']:.2f}, Max: {approx_stats['max']:.2f}, Mean: {approx_stats['mean']:.2f}, StdDev: {approx_stats['stddev']:.2f}, 'Median': {approx_stats['median']:.2f}, 'IQR: {approx_stats['iqr']:.2f}, Outliers: {approx_stats['outliers']}")

  print('\nTime statistics (seconds):')
  print(f"Min: {time_stats['min']:.4f}, Max: {time_stats['max']:.4f}, Mean: {time_stats['mean']:.4f}, StdDev: {time_stats['stddev']:.4f}, 'Median': {time_stats['median']:.4f}, IQR: {time_stats['iqr']:.4f}, Outliers: {time_stats['outliers']}")

  return approx_stats, time_stats


approx5, time5 = test_benchmark(5)
approx10, time10 = test_benchmark(10)
approx15, time15 = test_benchmark(15)