from ortools.constraint_solver import pywrapcp


class SearchMonitor(pywrapcp.SearchMonitor):
  def __init__(self, solver, q):
    pywrapcp.SearchMonitor.__init__(self, solver)
    self.q = q
    self.all_solutions = []
    self.unique_solutions = []
    self.count_symmetries = [0]*7
    self.n = len(self.q)

  def AcceptSolution(self):
    qval = [self.q[i].Value() for i in range(self.n)]
    self.all_solutions.append(qval)

    symmetries = [vv in self.unique_solutions for vv in gen_symmetries(self.n, qval)]
    self.count_symmetries = [i+v for i,v in zip(symmetries, self.count_symmetries)]

    if sum(symmetries) == 0:
      self.unique_solutions.append(qval)

    return False

def gen_symmetries(n, solution):

  symmetries = []

  x = list(range(n))
  for index in range(n):
    x[n - 1 - index] = solution[index]

  symmetries.append(x)

  #y(r[i]=j) → r[i]=n−j+1
  y = list(range(n))
  for index in range(n):
    y[index] = (n - 1 - solution[index])

  symmetries.append(y)

  #d1(r[i]=j) → r[j]=i
  d1 = list(range(n))
  for index in range(n):
    d1[solution[index]] = index

  symmetries.append(d1)

  # d2(r[i]=j) → r[n−j+1]=n−i+1
  d2 = list(range(n))
  for index in range(n):
    d2[n - 1 - solution[index]] = (n - 1 - index)

  symmetries.append(d2)

  # r90(r[i]=j) → r[j] = n−i+1
  r90 = list(range(n))
  for index in range(n):
    r90[solution[index]] = (n - 1 - index)

  symmetries.append(r90)

  # r180(r[i]=j) → r[n−i+1]=n−j+1
  r180 = list(range(n))
  for index in range(n):
    r180[n - 1 - index] = (n - 1 - solution[index])

  symmetries.append(r180)

  # r270(r[i]=j) → r[n−j+1]=i
  r270 = list(range(n))
  for index in range(n):
    r270[n - 1 - solution[index]] = index

  symmetries.append(r270)

  return symmetries

def n_queens(n):
  g_solver = pywrapcp.Solver("n-queens")
  q = [g_solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]

  g_solver.Add(g_solver.AllDifferent(q))
  g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
  g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))

  db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE_LOWEST_MAX,g_solver.ASSIGN_CENTER_VALUE)

  monitor = SearchMonitor(g_solver, q)
  g_solver.Solve(db, monitor)

  g_solver.NewSearch(db)

  while g_solver.NextSolution():
    pass

  g_solver.EndSearch()

  print("n: ", n)
  print("all_solutions:", len(monitor.all_solutions))
  print("unique_solutions:", len(monitor.unique_solutions))
  print("WallTime:", g_solver.WallTime(), "ms")

  return monitor

monitor = n_queens(5)
print("Unique Solutions: ", monitor.unique_solutions)
print("All Solutions: ", monitor.all_solutions)
