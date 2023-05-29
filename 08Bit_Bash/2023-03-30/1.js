// Computes the all 92 solutions to the eight queens problem
// by computing half of the results of the inductive solution 
// and then adding their vertical reflections.
//
// see http://raganwald.com/2018/08/03/eight-queens.html
//
// search space: 2,750 candidate positions

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

const allColumns = [0, 1, 2, 3, 4, 5, 6, 7];

function * halfInductive () {
  for (const column of [0, 1, 2, 3]) {
    const candidateQueens = [[0, column]];
    const remainingColumns = without(allColumns, column);
    yield * inductive(candidateQueens, remainingColumns);
  }
}

function verticalReflection (queens) {
  return queens.map(
    ([row, col]) => [row, 7 - col]
  );
}

function * flatMapWith (fn, iterable) {
  for (const element of iterable) {
    yield * fn(element);
  }
}

const withReflections = flatMapWith(
  queens => [queens, verticalReflection(queens)], halfInductive());

Array.from(withReflections).length
  //=> 92
