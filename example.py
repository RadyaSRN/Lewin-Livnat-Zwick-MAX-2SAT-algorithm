from lewin_livnat_zwick import *

n = 10

clauses_list = [(1, 2), (3, 4), (5, 6), (7, 8)]

total = ""
for pos1, pos2 in clauses_list:
  total = total + "("
  if pos1 > 0:
    total = total + f"x{pos1}"
  else:
    total = total + f"¬x{-pos1}"
  total = total + " v "
  if pos2 > 0:
    total = total + f"x{pos2}"
  else:
    total = total + f"¬x{-pos2}"
  total = total + ") ^ "
total = total[:-3]
print(f'Формула: {total}\n')

results = lewin_livnat_zwick(rotation, n, clauses_list)

print(f'Значения переменных:')
for i, res in enumerate(results):
  print(f'x{i}: {res}')