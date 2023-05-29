// Computes the all 92 solutions to the eight queens problem
// by testing partial solutions to the "rooks" algorithm
// as they are created, thus pruning subtrees when possible.
//
// see http://raganwald.com/2018/08/03/eight-queens.html
//
// search space: 5,508 candidate positions

function testDiagonals (queens) {
  const nesw = [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."];
  const nwse = [".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "."];

  if (queens.length < 2) return true;

  for (const [i, j] of queens) {
    if (nwse[i + j] !== '.' || nesw[i + 7 - j] !== '.') return false;

    nwse[i + j] = 'x';
    nesw[i + 7 - j] = 'x';
  }

  return true;
}

const without = (array, element) =>
	array.filter(x => x !== element);

function * inductive (
	queens = [],
  candidateColumns = [0, 1, 2, 3, 4, 5, 6, 7]
) {
  if (queens.length === 8) {
    yield queens;
  } else {
    for (const chosenColumn of candidateColumns) {
      const candidateQueens = queens.concat([[queens.length, chosenColumn]]);
      const remainingColumns = without(candidateColumns, chosenColumn);

      if (testDiagonals(candidateQueens)) {
        yield * inductive(candidateQueens, remainingColumns);
      }
    }
  }
}

Array.from(inductive()).length
  //=> 92
