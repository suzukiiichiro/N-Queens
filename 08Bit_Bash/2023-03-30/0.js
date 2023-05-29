// Computes the twelve "fundamental" solutions to the eight queens problem
// by filtering the results of the "half-inductive" algorithm.
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

const sortQueens = queens =>
  queens.reduce(
    (acc, [row, col]) => (acc[row] = [row, col], acc),
    [null, null, null, null, null, null, null, null]
  );

const rotateRight = queens =>
  sortQueens( queens.map(([row, col]) => [col, 7 - row]) );

const rotations = solution => {
  const rotations = [null, null, null];
  let temp = rotateRight(solution);

  rotations[0] = temp;
  temp = rotateRight(temp);
  rotations[1] = temp;
  temp = rotateRight(temp);
  rotations[2] = temp;

  return rotations;
}

const indexQueens = queens => queens.map(([row, col]) => `${row},${col}`).join(' ');

function * fundamentals (solutions) {
  const solutionsSoFar = new Set();

  for (const solution of solutions) {
    const iSolution = indexQueens(solution);

    if (solutionsSoFar.has(iSolution)) continue;

    solutionsSoFar.add(iSolution);
    const rSolutions = rotations(solution);
    const irSolutions = rSolutions.map(indexQueens);
    for (let irSolution of irSolutions) {
      solutionsSoFar.add(irSolution);
    }

    const vSolution = verticalReflection(solution);

    const rvSolutions = rotations(vSolution);
    const irvSolutions = rvSolutions.map(indexQueens);

    for (let irvSolution of irvSolutions) {
      solutionsSoFar.add(irvSolution);
    }

    yield solution;
  }
}

function * mapWith (mapFunction, iterable) {
  for (const element of iterable) {
    yield mapFunction(element);
  }
}

function niceDiagramOf (queens) {
  const board = [
    ["⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️"],
    ["⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️"],
    ["⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️"],
    ["⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️"],
    ["⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️"],
    ["⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️"],
    ["⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️"],
    ["⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️", "⬛️", "⬜️"]
  ];

  for (const [row, col] of queens) {
    board[7 - row][col] = "👸🏾";
  }

  return board.map(row => row.join('')).join("\n");
}

mapWith(niceDiagramOf, fundamentals(halfInductive()))
