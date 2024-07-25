import java.util.ArrayList;
import java.util.HashSet;
/**
 * 
 */
class Constellation{
  private int id;
  private int ld;
  private int rd;
  private int col;
  private int startijkl;
  private long solutions;
  public Constellation(){ super(); }
  public Constellation(int id, int ld, int rd, int col, int startijkl, long solutions){
    this.id = id;
    this.ld = ld;
    this.rd = rd;
    this.col = col;
    this.startijkl = startijkl;
    this.solutions = solutions;
  }
  public int getId(){ return id; }
  public void setId(int id){ this.id = id; }
  public int getLd(){ return ld; }
  public void setLd(int ld){ this.ld = ld; }
  public int getRd(){ return rd; }
  public void setRd(int rd){ this.rd = rd; }
  public int getCol(){ return col; }
  public void setCol(int col){ this.col = col; }
  public int getStartijkl(){ return startijkl; }
  public void setStartijkl(int startijkl){ this.startijkl = startijkl; }
  public long getSolutions(){ return solutions; }
  public void setSolutions(long solutions){ this.solutions = solutions; }
  public int getIjkl(){ return startijkl & 0b11111111111111111111; }
}
/**
 * 
 */
public class Main {
  private int L, mask, LD, RD, counter; 
  private int N=8;
  private int presetQueens;
  private HashSet<Integer> ijklList=new HashSet<Integer>();
  private static ArrayList<Constellation> constellations = new ArrayList<>(); 
  private long solutions, duration, storedDuration;
  private final int N3, N4, L3, L4; // boardsize
  private long tempcounter = 0; // tempcounter is #(unique solutions) of current start constellation, solvecounter is #(all solutions)
  private int mark1, mark2, endmark, jmark;
  //
  // generate subconstellations for each starting constellation with 3 or 4 queens
  private void setPreQueens(int ld, int rd, int col, int k, int l, int row, int queens){
    // in row k and l just go further
    if(row == k || row == l){
      setPreQueens(ld << 1, rd >>> 1, col, k, l, row + 1, queens);
      return;
    }
    // add queens until we have preQueens queens
    if(queens == presetQueens){
      // add the subconstellations to the list
      constellations.add(new Constellation(-1, ld, rd, col, row << 20, -1));
      counter++;
      return;
    }
    // if not done or row k or l, just place queens and occupy the board and go
    // further
    else {
      int free = (~(ld | rd | col | (LD >>> (N - 1 - row)) | (RD << (N - 1 - row)))) & mask;
      int bit;
      while(free>0){
        bit = free & (-free);
        free -= bit;
        setPreQueens((ld | bit) << 1, (rd | bit) >>> 1, col | bit, k, l, row + 1, queens + 1);
      }
    }
  }
  // true, if starting constellation rotated by any angle has already been found
  boolean checkRotations(HashSet<Integer> ijklList, int i, int j, int k, int l){
    // rot90
    if(ijklList.contains(((N - 1 - k) << 15) + ((N - 1 - l) << 10) + (j << 5) + i))
      return true;
    // rot180
    if(ijklList.contains(((N - 1 - j) << 15) + ((N - 1 - i) << 10) + ((N - 1 - l) << 5) + N - 1 - k))
      return true;
    // rot270
    if(ijklList.contains((l << 15) + (k << 10) + ((N - 1 - i) << 5) + N - 1 - j))
      return true;
    return false;
  }
  // i, j, k, l to ijkl and functions to get specific entry
  // rotate and mirror board, so that the queen closest to a corner is on the
  // right side of the last row
  int jasmin(int ijkl){
    int min = Math.min(getj(ijkl), N - 1 - getj(ijkl)), arg = 0;
    if(Math.min(geti(ijkl), N - 1 - geti(ijkl)) < min){
      arg = 2;
      min = Math.min(geti(ijkl), N - 1 - geti(ijkl));
    }
    if(Math.min(getk(ijkl), N - 1 - getk(ijkl)) < min){
      arg = 3;
      min = Math.min(getk(ijkl), N - 1 - getk(ijkl));
    }
    if(Math.min(getl(ijkl), N - 1 - getl(ijkl)) < min){
      arg = 1;
      min = Math.min(getl(ijkl), N - 1 - getl(ijkl));
    }
    for (int i = 0; i < arg; i++){
      ijkl = rot90(ijkl);
    }
    if(getj(ijkl) < N - 1 - getj(ijkl))
      ijkl = mirvert(ijkl);
    return ijkl;
  }
  // mirror left-right
  int mirvert(int ijkl){
    return toijkl(N - 1 - geti(ijkl), N - 1 - getj(ijkl), getl(ijkl), getk(ijkl));
  }
  // rotate 90 degrees clockwise
  int rot90(int ijkl){
    return ((N - 1 - getk(ijkl)) << 15) + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl);
  }
  // helper functions for doing the math
  // for symmetry stuff and working with ijkl
  // true, if starting constellation is symmetric for rot90
  boolean symmetry90(int ijkl){
    if(((geti(ijkl) << 15) + (getj(ijkl) << 10) + (getk(ijkl) << 5) + getl(ijkl)) == (((N - 1 - getk(ijkl)) << 15)
          + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl)))
      return true;
    return false;
  }
  // how often does a found solution count for this start constellation
  int symmetry(int ijkl){
    if(geti(ijkl) == N - 1 - getj(ijkl) && getk(ijkl) == N - 1 - getl(ijkl)) // starting constellation symmetric by rot180?
      if(symmetry90(ijkl)) // even by rot90?
        return 2;
      else
        return 4;
    else
      return 8; // none of the above?
  }
  //
  public long getSolutions(){
    return solutions;
  }
  private void calcSolutions(){
    for (var c : constellations){
      if(c.getSolutions() >= 0){
        solutions += c.getSolutions();
      }
    }
  }
  private void execSolutions(){
    int j, k, l, ijkl, ld, rd, col, startIjkl, start, free, LD;
    final int smallmask = (1 << (N - 2)) - 1;
    for (Constellation constellation : constellations){
      startIjkl = constellation.getStartijkl();
      start = startIjkl >> 20;
      ijkl = startIjkl & ((1 << 20) - 1);
      j = getj(ijkl);
      k = getk(ijkl);
      l = getl(ijkl);
      // IMPORTANT NOTE: we shift ld and rd one to the right, because the right
      // column does not matter (always occupied by queen l)
      // add occupation of ld from queens j and l from the bottom row upwards
      LD = (L >>> j) | (L >>> l);
      ld = constellation.getLd() >>> 1;
      ld |= LD >>> (N - start);
      // add occupation of rd from queens j and k from the bottom row upwards
      rd = constellation.getRd() >>> 1;
      if(start>k)
        rd |= (L >>> (start - k + 1));
      if(j >= 2 * N - 33 - start) // only add the rd from queen j if it does not
        rd |= (L >>> j) << (N - 2 - start); // occupy the sign bit!
      // also occupy col and then calculate free
      col = (constellation.getCol() >>> 1) | (~smallmask);
      free = ~(ld | rd | col);
      // big case distinction for deciding which soling algorithm to use
      // if queen j is more than 2 columns away from the corner
      if(j < N - 3){
        jmark = j + 1;
        endmark = N - 2;
        // if the queen j is more than 2 columns away from the corner but the rd from
        // the
        // j-queen can be set right at start
        if(j>2 * N - 34 - start){
          // k < l
          if(k < l){
            mark1 = k - 1;
            mark2 = l - 1;
            // if at least l is yet to come
            if(start < l){
              // if also k is yet to come
              if(start < k){
                // if there are free rows between k and l
                if(l != k + 1){
                  SQBkBlBjrB(ld, rd, col, start, free);
                }
                // if there are no free rows between k and l
                else {
                  SQBklBjrB(ld, rd, col, start, free);
                }
              }
              // if k already came before start and only l is left
              else {
                SQBlBjrB(ld, rd, col, start, free);
              }
            }
            // if both k and l already came before start
            else {
              SQBjrB(ld, rd, col, start, free);
            }
          }
          // l < k
          else {
            mark1 = l - 1;
            mark2 = k - 1;
            // if at least k is yet to come
            if(start < k){
              // if also l is yet to come
              if(start < l){
                // if there is at least one free row between l and k
                if(k != l + 1){
                  SQBlBkBjrB(ld, rd, col, start, free);
                }
                // if there is no free row between l and k
                else {
                  SQBlkBjrB(ld, rd, col, start, free);
                }
              }
              // if l already came and only k is yet to come
              else {
                SQBkBjrB(ld, rd, col, start, free);
              }
            }
            // if both l and k already came before start
            else {
              SQBjrB(ld, rd, col, start, free);
            }
          }
        }
        // if we have to set some queens first in order to reach the row N-1-jmark where
        // the
        // rd from queen j
        // can be set
        else {
          // k < l
          if(k < l){
            mark1 = k - 1;
            mark2 = l - 1;
            // there is at least one free row between rows k and l
            if(l != k + 1){
              SQBjlBkBlBjrB(ld, rd, col, start, free);
            }
            // if l comes right after k
            else {
              SQBjlBklBjrB(ld, rd, col, start, free);
            }
          }
          // l < k
          else {
            mark1 = l - 1;
            mark2 = k - 1;
            // there is at least on efree row between rows l and k
            if(k != l + 1){
              SQBjlBlBkBjrB(ld, rd, col, start, free);
            }
            // if k comes right after l
            else {
              SQBjlBlkBjrB(ld, rd, col, start, free);
            }
          }
        }
      }
      // if the queen j is exactly 2 columns away from the corner
      else if(j == N - 3){
        // this means that the last row will always be row N-2
        endmark = N - 2;
        // k < l
        if(k < l){
          mark1 = k - 1;
          mark2 = l - 1;
          // if at least l is yet to come
          if(start < l){
            // if k is yet to come too
            if(start < k){
              // if there are free rows between k and l
              if(l != k + 1){
                SQd2BkBlB(ld, rd, col, start, free);
              } else {
                SQd2BklB(ld, rd, col, start, free);
              }
            }
            // if k was set before start
            else {
              mark2 = l - 1;
              SQd2BlB(ld, rd, col, start, free);
            }
          }
          // if k and l already came before start
          else {
            SQd2B(ld, rd, col, start, free);
          }
        }
        // l < k
        else {
          mark1 = l - 1;
          mark2 = k - 1;
          endmark = N - 2;
          // if at least k is yet to come
          if(start < k){
            // if also l is yet to come
            if(start < l){
              // if there are free rows between l and k
              if(k != l + 1){
                SQd2BlBkB(ld, rd, col, start, free);
              }
              // if there are no free rows between l and k
              else {
                SQd2BlkB(ld, rd, col, start, free);
              }
            }
            // if l came before start
            else {
              mark2 = k - 1;
              SQd2BkB(ld, rd, col, start, free);
            }
          }
          // if both l and k already came before start
          else {
            SQd2B(ld, rd, col, start, free);
          }
        }
      }
      // if the queen j is exactly 1 column away from the corner
      else if(j == N - 2){
        // k < l
        if(k < l){
          // k can not be first, l can not be last due to queen placement
          // thus always end in line N-2
          endmark = N - 2;
          // if at least l is yet to come
          if(start < l){
            // if k is yet to come too
            if(start < k){
              mark1 = k - 1;
              // if k and l are next to each other
              if(l != k + 1){
                mark2 = l - 1;
                SQd1BkBlB(ld, rd, col, start, free);
              }
              //
              else {
                SQd1BklB(ld, rd, col, start, free);
              }
            }
            // if only l is yet to come
            else {
              mark2 = l - 1;
              SQd1BlB(ld, rd, col, start, free);
            }
          }
          // if k and l already came
          else {
            SQd1B(ld, rd, col, start, free);
          }
        }
        // l < k
        else {
          // if at least k is yet to come
          if(start < k){
            // if also l is yet to come
            if(start < l){
              // if k is not at the end
              if(k < N - 2){
                mark1 = l - 1;
                endmark = N - 2;
                // if there are free rows between l and k
                if(k != l + 1){
                  mark2 = k - 1;
                  SQd1BlBkB(ld, rd, col, start, free);
                }
                // if there are no free rows between l and k
                else {
                  SQd1BlkB(ld, rd, col, start, free);
                }
              }
              // if k is at the end
              else {
                // if l is not right before k
                if(l != N - 3){
                  mark2 = l - 1;
                  endmark = N - 3;
                  SQd1BlB(ld, rd, col, start, free);
                }
                // if l is right before k
                else {
                  endmark = N - 4;
                  SQd1B(ld, rd, col, start, free);
                }
              }
            }
            // if only k is yet to come
            else {
              // if k is not at the end
              if(k != N - 2){
                mark2 = k - 1;
                endmark = N - 2;
                SQd1BkB(ld, rd, col, start, free);
              } else {
                // if k is at the end
                endmark = N - 3;
                SQd1B(ld, rd, col, start, free);
              }
            }
          }
          // k and l came before start
          else {
            endmark = N - 2;
            SQd1B(ld, rd, col, start, free);
          }
        }
      }
      // if the queen j is placed in the corner
      else {
        endmark = N - 2;
        if(start>k){
          SQd0B(ld, rd, col, start, free);
        }
        // k can not be in the last row due to the way we construct start constellations
        // with a queen in the corner and
        // due to the way we apply jasmin
        else {
          mark1 = k - 1;
          SQd0BkB(ld, rd, col, start, free);
        }
      }
      // for saving and loading progress remove the finished starting constellation
      constellation.setSolutions(tempcounter * symmetry(ijkl));
      tempcounter = 0;
    }
  }
  private void genConstellations(){
    // halfN half of N rounded up
    final int halfN = (N + 1) / 2;
    L = 1 << (N - 1);
    mask = (1 << N) - 1;
    // calculate starting constellations for no Queens in corners
    for (int k = 1; k < halfN; k++){ // go through first col
      for (int l = k + 1; l < N - 1; l++){ // go through last col
        for (int i = k + 1; i < N - 1; i++){ // go through first row
          if(i == N - 1 - l) // skip if occupied
            continue;
          for (int j = N - k - 2; j>0; j--){ // go through last row
            if(j == i || l == j)
              continue;
            if(!checkRotations(ijklList, i, j, k, l)){ // if no rotation-symmetric starting
              // constellation already
              // found
              ijklList.add(toijkl(i, j, k, l));
            }
          }
        }
      }
    }
    // calculating start constellations with the first Queen on the corner square
    // (0,0)
    for (int j = 1; j < N - 2; j++){ // j is idx of Queen in last row
      for (int l = j + 1; l < N - 1; l++){ // l is idx of Queen in last col
        ijklList.add(toijkl(0, j, 0, l));
      }
    }
    HashSet<Integer> ijklListJasmin = new HashSet<Integer>();
    // rotate and mirror all start constellations, such that the queen in the last
    // row is as close to the right border as possible
    for (int startConstellation : ijklList){
      ijklListJasmin.add(jasmin(startConstellation));
    }
    ijklList = ijklListJasmin;
    int i, j, k, l, ld, rd, col, currentSize = 0;
    for (int sc : ijklList){
      i = geti(sc);
      j = getj(sc);
      k = getk(sc);
      l = getl(sc);
      // fill up the board with preQueens queens and generate corresponding variables
      // ld, rd, col, start_queens_ijkl for each constellation
      // occupy the board corresponding to the queens on the borders of the board
      // we are starting in the first row that can be free, namely row 1
      ld = (L >>> (i - 1)) | (1 << (N - k));
      rd = (L >>> (i + 1)) | (1 << (l - 1));
      col = 1 | L | (L >>> i) | (L >>> j);
      // occupy diagonals of the queens j k l in the last row
      // later we are going to shift them upwards the board
      LD = (L >>> j) | (L >>> l);
      RD = (L >>> j) | (1 << k);
      // counts all subconstellations
      counter = 0;
      // generate all subconstellations
      setPreQueens(ld, rd, col, k, l, 1, j == N - 1 ? 3 : 4);
      currentSize = constellations.size();
      // jkl and sym and start are the same for all subconstellations
      for (int a = 0; a < counter; a++){
        constellations.get(currentSize - a - 1)
          .setStartijkl(constellations.get(currentSize - a - 1).getStartijkl() | toijkl(i, j, k, l));
      }
    }
  }
  public Main(int sn){
    N=sn;
    presetQueens = 4;
    solutions=0;
    N3 = N - 3;
    N4 = N - 4;
    L = 1 << (N - 1);
    L3 = 1 << N3;
    L4 = 1 << N4;
  }   
  public static void main(String[] args){
    Main main = new Main(14);
    main.genConstellations();
    main.execSolutions();
    main.calcSolutions();
    System.out.println(main.getSolutions());
  }
  //
  int toijkl(int i, int j, int k, int l){ return (i << 15) + (j << 10) + (k << 5) + l; }
  int geti(int ijkl){ return ijkl >> 15; }
  int getj(int ijkl){ return (ijkl >> 10) & 31; }
  int getk(int ijkl){ return (ijkl >> 5) & 31; }
  int getl(int ijkl){ return ijkl & 31; }
  int getjkl(int ijkl){ return ijkl & 0b111111111111111; }
  //
  private void SQd0B(int ld, int rd, int col, int row, int free){
    if(row == endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      int next_ld = ((ld | bit) << 1);
      int next_rd = ((rd | bit) >> 1);
      int next_col = (col | bit);
      nextfree = ~(next_ld | next_rd | next_col);
      if(nextfree>0){
        if(row < endmark - 1){
          if(~((next_ld << 1) | (next_rd >> 1) | (next_col))>0)
            SQd0B(next_ld, next_rd, next_col, row + 1, nextfree);
        } else {
          SQd0B(next_ld, next_rd, next_col, row + 1, nextfree);
        }
      }
    }
  }
  private void SQd0BkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
        if(nextfree>0){
          SQd0B((ld | bit) << 2, ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd0BkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1BklB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | 1 | L4);
        if(nextfree>0){
          SQd1B(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BklB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1B(int ld, int rd, int col, int row, int free){
    if(row == endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      int next_ld = ((ld | bit) << 1);
      int next_rd = ((rd | bit) >> 1);
      int next_col = (col | bit);
      nextfree = ~(next_ld | next_rd | next_col);
      if(nextfree>0){
        if(row + 1 < endmark){
          if(~((next_ld << 1) | (next_rd >> 1) | (next_col))>0)
            SQd1B(next_ld, next_rd, next_col, row + 1, nextfree);
        } else {
          SQd1B(next_ld, next_rd, next_col, row + 1, nextfree);
        }
      }
    }
  }
  private void SQd1BkBlB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
        if(nextfree>0){
          SQd1BlB(((ld | bit) << 2), ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BkBlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1BlB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        int next_ld = ((ld | bit) << 2) | 1;
        int next_rd = ((rd | bit) >> 2);
        int next_col = (col | bit);
        nextfree = ~(next_ld | next_rd | next_col);
        if(nextfree>0){
          if(row + 2 < endmark){
            if(~((next_ld << 1) | (next_rd >> 1) | (next_col))>0)
              SQd1B(next_ld, next_rd, next_col, row + 2, nextfree);
          } else {
            SQd1B(next_ld, next_rd, next_col, row + 2, nextfree);
          }
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1BlkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | 2 | L3);
        if(nextfree>0){
          SQd1B(((ld | bit) << 3) | 2, ((rd | bit) >> 3) | L3, col | bit, row + 3, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BlkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1BlBkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if(nextfree>0){
          SQd1BkB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BlBkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd1BkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
        if(nextfree>0){
          SQd1B(((ld | bit) << 2), ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd1BkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BlkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L3 | 2);
        if(nextfree>0){
          SQd2B(((ld | bit) << 3) | 2, ((rd | bit) >> 3) | L3, col | bit, row + 3, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BlkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BklB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L4 | 1);
        if(nextfree>0){
          SQd2B(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BklB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BlBkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if(nextfree>0){
          SQd2BkB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BlBkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BkBlB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | (1 << (N3)));
        if(nextfree>0){
          SQd2BlB(((ld | bit) << 2), ((rd | bit) >> 2) | (1 << (N3)), col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BkBlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BlB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if(nextfree>0){
          SQd2B(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2BkB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
        if(nextfree>0){
          SQd2B(((ld | bit) << 2), ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQd2BkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQd2B(int ld, int rd, int col, int row, int free){
    if(row == endmark){
      if((free & (~1))>0){
        tempcounter++;
      }
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      int next_ld = ((ld | bit) << 1);
      int next_rd = ((rd | bit) >> 1);
      int next_col = (col | bit);
      nextfree = ~(next_ld | next_rd | next_col);
      if(nextfree>0){
        if(row < endmark - 1){
          if(~((next_ld << 1) | (next_rd >> 1) | (next_col))>0)
            SQd2B(next_ld, next_rd, next_col, row + 1, nextfree);
        } else {
          SQd2B(next_ld, next_rd, next_col, row + 1, nextfree);
        }
      }
    }
  }
  // for d>2 but d <small enough>
  private void SQBkBlBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | (1 << (N3)));
        if(nextfree>0){
          SQBlBjrB(((ld | bit) << 2), ((rd | bit) >> 2) | (1 << (N3)), col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBkBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBlBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if(nextfree>0){
          SQBjrB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == jmark){
      free &= (~1);
      ld |= 1;
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if(nextfree>0){
          SQB(((ld | bit) << 1), (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQB(int ld, int rd, int col, int row, int free){
    if(row == endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      int next_ld = ((ld | bit) << 1);
      int next_rd = ((rd | bit) >> 1);
      int next_col = (col | bit);
      nextfree = ~(next_ld | next_rd | next_col);
      if(nextfree>0){
        if(row < endmark - 1){
          if(~((next_ld << 1) | (next_rd >> 1) | (next_col))>0){
            SQB(next_ld, next_rd, next_col, row + 1, nextfree);
          }
        } else {
          SQB(next_ld, next_rd, next_col, row + 1, nextfree);
        }
      }
    }
  }
  private void SQBlBkBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if(nextfree>0){
          SQBkBjrB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBlBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBkBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark2){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
        if(nextfree>0){
          SQBjrB(((ld | bit) << 2), ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0)
        SQBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
    }
  }
  private void SQBklBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L4 | 1);
        if(nextfree>0)
          SQBjrB(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBklBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBlkBjrB(int ld, int rd, int col, int row, int free){
    int bit;
    int nextfree;
    if(row == mark1){
      while(free>0){
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L3 | 2);
        if(nextfree>0)
          SQBjrB(((ld | bit) << 3) | 2, ((rd | bit) >> 3) | L3, col | bit, row + 3, nextfree);
      }
      return;
    }
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBlkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  // for d <big>
  private void SQBjlBkBlBjrB(int ld, int rd, int col, int row, int free){
    if(row == N - 1 - jmark){
      rd |= L;
      free &= ~L;
      SQBkBlBjrB(ld, rd, col, row, free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBjlBkBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBjlBlBkBjrB(int ld, int rd, int col, int row, int free){
    if(row == N - 1 - jmark){
      rd |= L;
      free &= ~L;
      SQBlBkBjrB(ld, rd, col, row, free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBjlBlBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBjlBklBjrB(int ld, int rd, int col, int row, int free){
    if(row == N - 1 - jmark){
      rd |= L;
      free &= ~L;
      SQBklBjrB(ld, rd, col, row, free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBjlBklBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
  private void SQBjlBlkBjrB(int ld, int rd, int col, int row, int free){
    if(row == N - 1 - jmark){
      rd |= L;
      free &= ~L;
      SQBlkBjrB(ld, rd, col, row, free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit = free & (-free);
      free -= bit;
      nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
      if(nextfree>0){
        SQBjlBlkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
      }
    }
  }
}
