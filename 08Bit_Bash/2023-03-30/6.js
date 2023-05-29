
const OCCUPATION_HELPER = Symbol("occupationHelper");

class Board {
  constructor () {
    this.threats = [
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0
    ];
    this.queenIndices = [];
  }
  
  isValid (index) {
    return index >= 0 && index <= 63;
  }
  
  isAvailable (index) {
    return this.threats[index] === 0;
  }
  
  isEmpty () {
    return this.queenIndices.length === 0;
  }
  
  isOccupiable (index) {
    if (this.isEmpty()) {
      return this.isValid(index);
    } else {
      return this.isValid(index) && index > this.lastQueen() && this.isAvailable(index);
    }
  }
  
  numberOfQueens () {
    return this.queenIndices.length
  }
  
  hasQueens () {
    return this.numberOfQueens() > 0;
  }
  
  queens () {
    return this.queenIndices.map(index => [Math.floor(index / 8), index % 8]);
  }

  lastQueen () {
    if (this.queenIndices.length > 0) {
      return this.queenIndices[this.queenIndices.length - 1];
    }
  }

  * availableIndices () {
    for (let index = (this.isEmpty() ? 0 : this.lastQueen() + 1); index <= 63; ++index) {
      if (this.isAvailable(index)) {
        yield index;
      }
    }
  }
  
  [OCCUPATION_HELPER] (index, action) {
    const [row, col] = [Math.floor(index / 8), index % 8];
    
    // the rest of the row
    const endOfTheRow = row * 8 + 7;
    for (let iThreatened = index + 1; iThreatened <= endOfTheRow; ++iThreatened) {
      action(iThreatened);
    }
    
    // the rest of the column
    const endOfTheColumn = 56 + col;
    for (let iThreatened = index + 8; iThreatened <= endOfTheColumn; iThreatened += 8) {
      action(iThreatened);
    }
    
    // diagonals to the left
    const lengthOfLeftDiagonal = Math.min(col, 7 - row);
    for (let i = 1; i <= lengthOfLeftDiagonal; ++i) {
      const [rowThreatened, colThreatened] = [row + i, col - i];
      const iThreatened = rowThreatened * 8 + colThreatened;

      action(iThreatened);
    }
    
    // diagonals to the right
    const lengthOfRightDiagonal = Math.min(7 - col, 7 - row);
    for (let i = 1; i <= lengthOfRightDiagonal; ++i) {
      const [rowThreatened, colThreatened] = [row + i, col + i];
      const iThreatened = rowThreatened * 8 + colThreatened;

      action(iThreatened);
    }
    
    return this;
  }
  
  occupy (index) {
    const occupyAction = index => {
      ++this.threats[index];
    };
    
    if (this.isOccupiable(index)) {
      this.queenIndices.push(index);
      return this[OCCUPATION_HELPER](index, occupyAction);
    }
  }
  
  unoccupy () {
    const unoccupyAction = index => {
      --this.threats[index];
    };
    
    if (this.hasQueens()) {
      const index = this.queenIndices.pop();
      
      return this[OCCUPATION_HELPER](index, unoccupyAction);
    }
  }
}

function * inductive (board = new Board()) {
  if (board.numberOfQueens() === 8) {
    yield board.queens();
  } else {
    for (const index of board.availableIndices()) {
      board.occupy(index);
      yield * inductive(board);
      board.unoccupy();
    }
  }
}
