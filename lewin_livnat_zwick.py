import cvxpy as cp
import numpy as np
import scipy
import scipy.stats as sps

def rotation(x):
  """
  Функция вращения из статьи
  
  :param x: аргумент функции
  :return: значение функции в x
  """

  return 0.58831458 * x + 0.64667394
  

def rotate_vector(rotation_func, v_0, v_i):
  """
  Делает поворот v_i

  :param rotation_func: функция, вычисляющая угол, который будет после поворота
  :param v_0: поворот делается в плоскости, образованной v_0 и v_i
  :param v_i: вектор, который поворачивается
  :return: повернутый вектор
  """

  angle_cos = v_0 @ v_i

  # Вычисляем проекцию v_i и на v_0, а также единичный перпендикуляр от нее к v_i
  projection = angle_cos * v_0
  perpendicular = v_i - projection
  perpendicular /= np.linalg.norm(perpendicular)

  # Вычисляем новый угол, вычисляем новый вектор через проекцию и перпендикуляр
  new_angle = rotation_func(np.arccos(np.clip(angle_cos, -1, 1)))
  new_projection = np.cos(new_angle) * v_0
  new_perpendicular = np.sin(new_angle) * perpendicular

  return new_projection + new_perpendicular


def lewin_livnat_zwick(rotation_func, n, clauses_list):
  """
  Имплементация алгоритма Левина-Ливната-Цвика

  :param rotation_func: функция, вычисляющая угол, который будет после поворота
  :param n: число переменных в задаче
  :param clauses_list: список из 2n элементов, обозначающих элементы дизъюнктов
  :return: булевы значения x_0, ..., x_n
  """ 
  # Вычисляем исходную матрицу весов
  weight_matrix = np.zeros((2 * n + 1, 2 * n + 1))
  for i in range(0, len(clauses_list), 2):
    weight_matrix[clauses_list[i], clauses_list[i + 1]] = 1

  # Вычисляем матрицу весов в задаче
  weight_matrix_modified = weight_matrix.copy()
  weight_matrix_modified_first_column_sum = np.sum(weight_matrix, axis=1)
  weight_matrix_modified_first_row_sum = np.sum(weight_matrix, axis=0)
  weight_matrix_modified[0, :] += weight_matrix_modified_first_column_sum
  weight_matrix_modified[0, :] += weight_matrix_modified_first_row_sum
  C = -weight_matrix_modified / 4

  # Ограничения вида неравенств
  A_inequality = []
  for i in range(1, 2 * n + 1):
    for j in range(1, 2 * n + 1):
      current_A = np.zeros((2 * n + 1, 2 * n + 1))
      current_A[0, i] = 1
      current_A[0, j] = 1
      current_A[i, j] = 1
      A_inequality.append(current_A)
  b_inequality = [-1] * len(A_inequality)

  # Ограничения на противоположность знаков v_i и v_{n + i}
  A_equality_minus_one = []
  for i in range(1, n + 1):
    current_A = np.zeros((2 * n + 1, 2 * n + 1))
    current_A[i, n + i] = 1
    A_equality_minus_one.append(current_A)
  b_equality_minus_one = [-1] * len(A_equality_minus_one)

  # Ограничения на единичную норму v_i
  A_equality_one = []
  for i in range(0, 2 * n + 1):
    current_A = np.zeros((2 * n + 1, 2 * n + 1))
    current_A[i, i] = 1
    A_equality_one.append(current_A)
  b_equality_one = [1] * len(A_equality_one)

  # Переменная задачи
  X = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)

  # Записываем все ограничения
  constraints = [X >> 0]
  constraints += [
      cp.trace(A_inequality[i].T @ X) >= b_inequality[i] for i in range(len(A_inequality))
  ]
  constraints += [
      cp.trace(A_equality_minus_one[i].T @ X) == b_equality_minus_one[i] for i in range(len(A_equality_minus_one))
  ]
  constraints += [
      cp.trace(A_equality_one[i].T @ X) == b_equality_one[i] for i in range(len(A_equality_one))
  ]

  # Решаем задачу
  prob = cp.Problem(cp.Maximize(cp.trace(C.T @ X)), constraints)
  prob.solve()

  # Получаем вектора по матрице X
  eigenvalues, eigenvectors = np.linalg.eigh(X.value)
  vectors_transposed = np.sqrt(np.abs(eigenvalues)).reshape(-1, 1) * eigenvectors.T
  vectors = vectors_transposed.T

  # Получаем через QR-разложение вид, в котором v_0 = (1 0 ... 0)
  Q, R = scipy.linalg.qr(vectors.T)
  if R[0, 0] < 0:
    R = -R
  vectors_transformed = R.T
  vectors_transformed_squeezed = vectors_transformed[:n + 1, :n + 1]

  # Поворачиваем вектора
  vectors_rotated = np.vstack((
      np.array([vectors_transformed_squeezed[0]]),
      np.array([rotate_vector(rotation_func, vectors_transformed_squeezed[0], vectors_transformed_squeezed[i]) for i in range(1, n + 1)])
  ))

  # Делаем округления гиперплоскостью
  plane_normal_vector = np.hstack(([2], sps.norm.rvs(size=n)))
  results = [(vectors_rotated[0] @ plane_normal_vector) * (vectors_rotated[i] @ plane_normal_vector) <= 0 for i in range(n + 1)]

  return results