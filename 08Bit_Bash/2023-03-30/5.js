// Computes the all 92 solutions to the eight queens problem
// by computing all of the possible arrangements of eight
// chess squares (64^8), then filtering them
// for horizontal, vertical, and diagonal attacks.
//
// see http://raganwald.com/2018/08/03/eight-queens.html
//
// search space: 281,474,976,710,656 candidate positions

function diagramOf (queens) {
  const board = [
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."]
  ];

  for (const [i, j] of queens) {
    board[i][j] = "Q";
  }

  return board.map(row => row.join('')).join("\n");
}

function * mostPessimumGenerator () {
  for (let i0 = 0; i0 <= 7; ++i0) {
    for (let j0 = 0; j0 <= 7; ++j0) {
      for (let i1 = 0; i1 <= 7; ++i1) {
        for (let j1 = 0; j1 <= 7; ++j1) {
          for (let i2 = 0; i2 <= 7; ++i2) {
            for (let j2 = 0; j2 <= 7; ++j2) {
              for (let i3 = 0; i3 <= 7; ++i3) {
                for (let j3 = 0; j3 <= 7; ++j3) {
                  for (let i4 = 0; i4 <= 7; ++i4) {
                    for (let j4 = 0; j4 <= 7; ++j4) {
                      for (let i5 = 0; i5 <= 7; ++i5) {
                        for (let j5 = 0; j5 <= 7; ++j5) {
                          for (let i6 = 0; i6 <= 7; ++i6) {
                            for (let j6 = 0; j6 <= 7; ++j6) {
                              for (let i7 = 0; i7 <= 7; ++i7) {
                                for (let j7 = 0; j7 <= 7; ++j7) {
                                  const queens = [
                                    [i0, j0],
                                    [i1, j1],
                                    [i2, j2],
                                    [i3, j3],
                                    [i4, j4],
                                    [i5, j5],
                                    [i6, j6],
                                    [i7, j7]
                                  ];

                                  yield queens;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

function test (queens) {
  const board = [
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."]
  ];

  for (const [i, j] of queens) {
    if (board[i][j] != '.') {
      // square is occupied or threatened
      return false;
    }

    for (let k = 0; k <= 7; ++k) {
      // fill row and column
      board[i][k] = board[k][j] = "x";

      const vOffset = k-i;
      const hDiagonal1 = j - vOffset;
      const hDiagonal2 = j + vOffset;

      // fill diagonals
      if (hDiagonal1 >= 0 && hDiagonal1 <= 7) {
        board[k][hDiagonal1] = "x";
      }

      if (hDiagonal2 >= 0 && hDiagonal2 <= 7) {
        board[k][hDiagonal2] = "x";
      }

      board[i][j] = "Q";
    }
  }

  return true;
}

function * filterWith (predicateFunction, iterable) {
  for (const element of iterable) {
    if (predicateFunction(element)) {
      yield element;
    }
  }
}

function first (iterable) {
  const [value] = iterable;

  return value;
}

const solutionsToEightQueens = filterWith(test, mostPessimumGenerator());

diagramOf(first(solutionsToEightQueens))
