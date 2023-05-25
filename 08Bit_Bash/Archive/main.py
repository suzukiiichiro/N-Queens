from n_queens import NQueens
from datetime import datetime


def main():
    print('.: N-Queens Problem :.')
    size = int(input('Please enter the size of board: '))
    print_solutions = input('Do you want the solutions to be printed (Y/N): ').lower() == 'y'
    n_queens = NQueens(size)
    #start_time = datetime.now()
    #dfs_solutions = n_queens.solve_dfs()
    #time_elapsed_dfs = datetime.now() - start_time
    start_time = datetime.now()
    bfs_solutions = n_queens.solve_bfs()
    time_elapsed_bfs = datetime.now() - start_time
    #if print_solutions:
    #    for i, solution in enumerate(dfs_solutions):
    #        print('DFS Solution %d:' % (i + 1))
    #        n_queens.print(solution)
    #    for i, solution in enumerate(bfs_solutions):
    #        print('BFS Solution %d:' % (i + 1))
    #        n_queens.print(solution)
    #_text = '{}'.format(time_elapsed_dfs)
    #text = _text[:-3]
    #print('Total DFS solutions: %d:%20s' % (len(dfs_solutions),text))
    #_text = '{}'.format(time_elapsed_bfs)
    #text = _text[:-3]
    #print('Total BFS solutions: %d:%20s' % (len(bfs_solutions),text))


if __name__ == '__main__':
    main()
