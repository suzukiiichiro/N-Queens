// Computes the all 92 solutions to the eight queens problem
// by computing all of the ways eight queens can be arranged
// on the board using 64 choose 8, then filtering them
// for horizontal, vertical, and diagonal attacks.
//
// see http://raganwald.com/2018/08/03/eight-queens.html
//
// search space: 4,426,165,368 candidate positions

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

function * mapWith (mapFunction, iterable) {
  for (const element of iterable) {
    yield mapFunction(element);
  }
}

function * choose (n, k, offset = 0) {
  if (k === 1) {
    for (let i = 0; i <= (n - k); ++i) {
      yield [i + offset];
    }
  } else if (k > 1) {
    for (let i = 0; i <= (n - k); ++i) {
      const remaining = n - i - 1;
      const otherChoices = choose(remaining, k - 1, i + offset + 1);

      yield * mapWith(x => [i + offset].concat(x), otherChoices);
    }
  }
}

const numberToPosition = n => [Math.floor(n/8), n % 8];
const numbersToPositions = queenNumbers => queenNumbers.map(numberToPosition);

const combinationCandidates = mapWith(numbersToPositions, choose(64, 8));

const solutionsToEightQueens = filterWith(test, combinationCandidates);

diagramOf(first(solutionsToEightQueens))
