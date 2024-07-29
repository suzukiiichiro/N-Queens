import java.util.ArrayList;
import java.util.HashSet;
/**
 * 
 */
class Constellation
{
  private int id;
  private int ld;
  private int rd;
  private int col;
  private int startijkl;
  private long solutions;
  public Constellation(){ super();}
  public Constellation(int id,int ld,int rd,int col,int startijkl,long solutions){
    this.id=id;
    this.ld=ld;
    this.rd=rd;
    this.col=col;
    this.startijkl=startijkl;
    this.solutions=solutions;
  }
  public int getId(){ return id;}
  public void setId(int id){ this.id=id;}
  public int getLd(){ return ld;}
  public void setLd(int ld){ this.ld=ld;}
  public int getRd(){ return rd;}
  public void setRd(int rd){ this.rd=rd;}
  public int getCol(){ return col;}
  public void setCol(int col){ this.col=col;}
  public int getStartijkl(){ return startijkl;}
  public void setStartijkl(int startijkl){ this.startijkl=startijkl;}
  public long getSolutions(){ return solutions;}
  public void setSolutions(long solutions){ this.solutions=solutions;}
  public int getIjkl(){ return startijkl&0b11111111111111111111;}
}
/**
 * 
 */
public class Main
{
  private int L,mask,LD,RD,counter;
  private int N=8;
  private int presetQueens;
  private HashSet<Integer> ijklList=new HashSet<Integer>();
  private static ArrayList<Constellation> constellations=new ArrayList<>();
  private long solutions,duration,storedDuration;
  private final int N3,N4,L3,L4;// ボードサイズ
  // tempcounter is #(unique solutions) of current start constellation,solvecounter is #(all solutions)
  // tempcounterは現在の開始座標の#(ユニークな解)、solvecounterは#(すべての解)
  private long tempcounter=0;
  private int mark1,mark2,endmark,jmark;
  //
  // 3つまたは4つのクイーンを使って開始コンステレーションごとにサブコンステレーションを生成する。
  private void setPreQueens(int ld,int rd,int col,int k,int l,int row,int queens)
  {
    // k行とl行はさらに進む
    if(row==k || row==l){
      setPreQueens(ld<<1,rd>>>1,col,k,l,row+1,queens);
      return;
    }
    // preQueensのクイーンが揃うまでクイーンを追加する。
    if(queens==presetQueens){
      // リストにサブカテゴリーを追加する
      constellations.add(new Constellation(-1,ld,rd,col,row<<20,-1));
      counter++;
      return;
    }
    // k列かl列が終わっていなければ、クイーンを置いてボードを占領し、さらに先に進む。
   else{
      int free=(~(ld|rd|col|(LD>>>(N-1-row))|(RD<<(N-1-row))))&mask;
      int bit;
      while(free>0){
        bit=free&(-free);
        free-=bit;
        setPreQueens((ld|bit)<<1,(rd|bit)>>>1,col|bit,k,l,row+1,queens+1);
      }
    }
  }
  // いずれかの角度で回転させた星座がすでに見つかっている場合、trueを返す。
  boolean checkRotations(HashSet<Integer> ijklList,int i,int j,int k,int l)
  {
    // rot90
    if(ijklList.contains(((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i))
      return true;
    // rot180
    if(ijklList.contains(((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+N-1-k))
      return true;
    // rot270
    if(ijklList.contains((l<<15)+(k<<10)+((N-1-i)<<5)+N-1-j))
      return true;
    return false;
  }
  // i,j,k,lをijklに変換し、特定のエントリーを取得する関数 
  // コーナーに最も近いクイーンが最後の列の右側になるようにボードを回転させミラーにする。
  int jasmin(int ijkl)
  {
    int min=Math.min(getj(ijkl),N-1-getj(ijkl)),arg=0;
    if(Math.min(geti(ijkl),N-1-geti(ijkl))<min){
      arg=2;
      min=Math.min(geti(ijkl),N-1-geti(ijkl));
    }
    if(Math.min(getk(ijkl),N-1-getk(ijkl))<min){
      arg=3;
      min=Math.min(getk(ijkl),N-1-getk(ijkl));
    }
    if(Math.min(getl(ijkl),N-1-getl(ijkl))<min){
      arg=1;
      min=Math.min(getl(ijkl),N-1-getl(ijkl));
    }
    for (int i=0;i<arg;i++){
      ijkl=rot90(ijkl);
    }
    if(getj(ijkl)<N-1-getj(ijkl))
      ijkl=mirvert(ijkl);
    return ijkl;
  }
  // 左右のミラー
  int mirvert(int ijkl)
  {
    return toijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl));
  }
  // 時計回りに90度回転
  int rot90(int ijkl)
  {
    return ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl);
  }
  // 対称性のための計算と、ijklを扱うためのヘルパー関数。
  // 開始コンステレーションが回転90に対して対称である場合
  boolean symmetry90(int ijkl)
  {
    if(((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl))==(((N-1-getk(ijkl))<<15)
         +((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl)))
      return true;
    return false;
  }
  // この開始コンステレーションで、見つかった解がカウントされる頻度
  int symmetry(int ijkl)
  {
    // コンステレーションをrot180で対称に開始するか？
    if(geti(ijkl)==N-1-getj(ijkl) && getk(ijkl)==N-1-getl(ijkl)){
      if(symmetry90(ijkl)) // 90？
        return 2;
      else
        return 4;
    }else{
      return 8;// 上記のどれでもない？
    }
  }
  //
  public long getSolutions()
  {
    return solutions;
  }
  private void calcSolutions()
  {
    for (var c : constellations){
      if(c.getSolutions() >= 0){
        solutions += c.getSolutions();
      }
    }
  }
  private void execSolutions()
  {
    int j,k,l,ijkl,ld,rd,col,startIjkl,start,free,LD;
    final int smallmask=(1<<(N-2))-1;
    for (Constellation constellation : constellations){
      startIjkl=constellation.getStartijkl();
      start=startIjkl>>20;
      ijkl=startIjkl&((1<<20)-1);
      j=getj(ijkl);
      k=getk(ijkl);
      l=getl(ijkl);
      // 重要な注意：ldとrdを1つずつ右にずらすが、これは右列は重要ではないから（常に女王lが占有している）。
      // 最下段から上に、jとlのクイーンによるldの占有を追加する。
      LD=(L>>>j)|(L>>>l);
      ld=constellation.getLd()>>>1;
      ld|=LD>>>(N-start);
      // クイーンjとkのrdの占有率を下段から上に加算する。
      rd=constellation.getRd()>>>1;
      if(start>k)
        rd|=(L>>>(start-k+1));
      // クイーンjからのrdがない場合のみ追加する
      if(j >= 2 * N-33-start){
        // 符号ビットを占有する！
        rd|=(L>>>j)<<(N-2-start);
      }
      // また、colを占有し、次にフリーを計算する
      col=(constellation.getCol()>>>1)|(~smallmask);
      free=~(ld|rd|col);
      // どのソリングアルゴリズムを使うかを決めるための大きなケースの区別
      // クイーンjがコーナーから2列以上離れている場合
      if(j<N-3){
        jmark=j+1;
        endmark=N-2;
        // クイーンjがコーナーから2列以上離れているが、jクイーンからのrdが開始時に正しく設定できる場合。
        if(j>2 * N-34-start){
          // k<l
          if(k<l){
            mark1=k-1;
            mark2=l-1;
            // 少なくともlがまだ来ていない場合
            if(start<l){
              // もしkがまだ来ていないなら
              if(start<k){
                // kとlの間に空行がある場合
                if(l != k+1){
                  SQBkBlBjrB(ld,rd,col,start,free);
                }
                // kとlの間に空行がない場合
               else{
                  SQBklBjrB(ld,rd,col,start,free);
                }
              }
              // もしkがすでに開始前に来ていて、lだけが残っている場合
             else{
                SQBlBjrB(ld,rd,col,start,free);
              }
            }
            // kとlの両方が開始前にすでに来ていた場合
           else{
              SQBjrB(ld,rd,col,start,free);
            }
          }
          // l<k
         else{
            mark1=l-1;
            mark2=k-1;
            // 少なくともkがまだ来ていない場合
            if(start<k){
              // if also l is yet to come
              // lがまだ来ていない場合
              if(start<l){
                // lとkの間に少なくとも1つの自由行がある場合
                if(k != l+1){
                  SQBlBkBjrB(ld,rd,col,start,free);
                }
                // lとkの間に自由行がない場合
               else{
                  SQBlkBjrB(ld,rd,col,start,free);
                }
              }
              // lがすでに来ていて、kだけがまだ来ていない場合
             else{
                SQBkBjrB(ld,rd,col,start,free);
              }
            }
            // lとkの両方が開始前にすでに来ていた場合
           else{
              SQBjrB(ld,rd,col,start,free);
            }
          }
        }
        // クイーンjのrdをセットできる行N-1-jmarkに到達するために、最初にいくつかのクイーンをセットしなければならない場合。
       else{
          // k<l
          if(k<l){
            mark1=k-1;
            mark2=l-1;
            // k行とl行の間に少なくとも1つの空行がある。
            if(l != k+1){
              SQBjlBkBlBjrB(ld,rd,col,start,free);
            }
            // lがkの直後に来る場合
           else{
              SQBjlBklBjrB(ld,rd,col,start,free);
            }
          }
          // l<k
         else{
            mark1=l-1;
            mark2=k-1;
            // l行とk行の間には、少なくともefree行が存在する。
            if(k != l+1){
              SQBjlBlBkBjrB(ld,rd,col,start,free);
            }
            // kがlの直後に来る場合
           else{
              SQBjlBlkBjrB(ld,rd,col,start,free);
            }
          }
        }
      }else if(j==N-3){
      // クイーンjがコーナーからちょうど2列離れている場合
        // これは、最終行が常にN-2行になることを意味する。
        endmark=N-2;
        // k<l
        if(k<l){
          mark1=k-1;
          mark2=l-1;
          // 少なくともlがまだ来ていない場合
          if(start<l){
            // もしkもまだ来ていないなら
            if(start<k){
              // kとlの間に空行がある場合
              if(l != k+1){
                SQd2BkBlB(ld,rd,col,start,free);
              }else{
                SQd2BklB(ld,rd,col,start,free);
              }
            }
            // k が開始前に設定されていた場合
           else{
              mark2=l-1;
              SQd2BlB(ld,rd,col,start,free);
            }
          }
          // もしkとlが開始前にすでに来ていた場合
         else{
            SQd2B(ld,rd,col,start,free);
          }
        }
        // l<k
        else{
          mark1=l-1;
          mark2=k-1;
          endmark=N-2;
          // 少なくともkがまだ来ていない場合
          if(start<k){
            // lがまだ来ていない場合
            if(start<l){
              // lとkの間に空行がある場合
              if(k != l+1){
                SQd2BlBkB(ld,rd,col,start,free);
              }
              // lとkの間に空行がない場合
             else{
                SQd2BlkB(ld,rd,col,start,free);
              }
            }
            // l が開始前に来た場合
           else{
              mark2=k-1;
              SQd2BkB(ld,rd,col,start,free);
            }
          }
          // lとkの両方が開始前にすでに来ていた場合
         else{
            SQd2B(ld,rd,col,start,free);
          }
        }
      }else if(j==N-2){
      // クイーンjがコーナーからちょうど1列離れている場合
        // k<l
        if(k<l){
          // kが最初になることはない、lはクイーンの配置の関係で最後尾にはなれないので、常にN-2行目で終わる。
          endmark=N-2;
          // 少なくともlがまだ来ていない場合
          if(start<l){
            // もしkもまだ来ていないなら
            if(start<k){
              mark1=k-1;
              // kとlが隣り合っている場合
              if(l != k+1){
                mark2=l-1;
                SQd1BkBlB(ld,rd,col,start,free);
              }
              //
             else{
                SQd1BklB(ld,rd,col,start,free);
              }
            }
            // lがまだ来ていないなら
           else{
              mark2=l-1;
              SQd1BlB(ld,rd,col,start,free);
            }
          }
          // すでにkとlが来ている場合
         else{
            SQd1B(ld,rd,col,start,free);
          }
        }
        // l<k
       else{
          // 少なくともkがまだ来ていない場合
          if(start<k){
            // if also l is yet to come
            // lがまだ来ていない場合
            if(start<l){
              // kが末尾にない場合
              if(k<N-2){
                mark1=l-1;
                endmark=N-2;
                // lとkの間に空行がある場合
                if(k != l+1){
                  mark2=k-1;
                  SQd1BlBkB(ld,rd,col,start,free);
                }
                // lとkの間に空行がない場合
               else{
                  SQd1BlkB(ld,rd,col,start,free);
                }
              }
              // kが末尾の場合
             else{
                // lがkの直前でない場合
                if(l != N-3){
                  mark2=l-1;
                  endmark=N-3;
                  SQd1BlB(ld,rd,col,start,free);
                }
                // lがkの直前にある場合
               else{
                  endmark=N-4;
                  SQd1B(ld,rd,col,start,free);
                }
              }
            }
            // もしkがまだ来ていないなら
           else{
              // kが末尾にない場合
              if(k != N-2){
                mark2=k-1;
                endmark=N-2;
                SQd1BkB(ld,rd,col,start,free);
              }else{
                // kが末尾の場合
                endmark=N-3;
                SQd1B(ld,rd,col,start,free);
              }
            }
          }else{
          // kとlはスタートの前
            endmark=N-2;
            SQd1B(ld,rd,col,start,free);
          }
        }
      }else{
      // クイーンjがコーナーに置かれている場合
        endmark=N-2;
        if(start>k){
          SQd0B(ld,rd,col,start,free);
        }
        // クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用する方法によって、Kは最後列に入ることはできない。
       else{
          mark1=k-1;
          SQd0BkB(ld,rd,col,start,free);
        }
      }
      // 完成した開始コンステレーションを削除する。
      constellation.setSolutions(tempcounter * symmetry(ijkl));
      tempcounter=0;
    }
  }
  private void genConstellations()
  {
    // N の半分を切り上げる
    final int halfN=(N+1) / 2;
    //Lは左端に1を立てる
    L=1<<(N-1);
    //maskはNビットの全てが1のビットマスクです
    mask=(1<<N)-1;
    // コーナーにクイーンがいない場合の開始コンステレーションを計算する
    // 最初のcolを通過する
    //k: 最初の列（左端）に配置されるクイーンの行のインデックス。
    for (int k=1;k<halfN;k++){
      //l: 最後の列（右端）に配置されるクイーンの行のインデックス。
      //l を k より後の行に配置する理由は、回転対称性を考慮して配置の重複を避けるためです。
      //このアプローチにより、探索空間が効率化され、N-クイーン問題の解決が迅速かつ効率的に行えるようになります。
      // 最後のcolを通過する
      for (int l=k+1;l<N-1;l++){
        //i: 最初の行（上端）に配置されるクイーンの列のインデックス。
        // 最初の行を通過する
        //k よりも下の行に配置することで、ボード上の対称性や回転対称性を考慮して、重複した解を避けるための配慮がされています。
        for (int i=k+1;i<N-1;i++){
          // i==N-1-lは、行iが列lの「対角線上」にあるかどうかをチェックしています。
          if(i==N-1-l){
            continue;
          }
          //j: 最後の行（下端）に配置されるクイーンの列のインデックス。  
          // 最後の行を通過する
          for (int j=N-k-2;j>0;j--){
            //同じ列や行にクイーンが配置されている場合は、その配置が有効でないためスキップ
            if(j==i || l==j){
              continue;
            }
            // 回転対称でスタートしない場合
            //checkRotationsで回転対称性をチェックし、対称でない場合にijklListに配置を追加します。
            if(!checkRotations(ijklList,i,j,k,l)){
              // すでにコンステレーションが見つかった
              ijklList.add(toijkl(i,j,k,l));
            }
          }
        }
      }
    }
    // 最初のクイーンを角の正方形に置いてスタートコンステレーションを計算する
    // (0,0)
    for (int j=1;j<N-2;j++){ // j is idx of Queen in last row
      for (int l=j+1;l<N-1;l++){ // l is idx of Queen in last col
        ijklList.add(toijkl(0,j,0,l));
      }
    }
    HashSet<Integer> ijklListJasmin=new HashSet<Integer>();
    // すべての開始星座を回転させ、ミラーリングする。
    // 最後の行のクイーンができるだけ右のボーダーに近づくようにする。
    for (int startConstellation : ijklList){
      //jasmin関数を使用して、全ての開始コンステレーションを回転・ミラーリングします。
      ijklListJasmin.add(jasmin(startConstellation));
    }
    ijklList=ijklListJasmin;
    int i,j,k,l,ld,rd,col,currentSize=0;
    for (int sc : ijklList){
      i=geti(sc);
      j=getj(sc);
      k=getk(sc);
      l=getl(sc);
      // プレクイーンでボードを埋め、対応する変数を生成する。
      // 各星座に対して ld,rd,col,start_queens_ijkl を設定する。
      // 碁盤の境界線上のクイーンに対応する碁盤を占有する。
      // 空いている最初の行、すなわち1行目から開始する。
      //クイーンの左対角線上の攻撃範囲を設定する。
      //L>>>(i-1) は、Lを (i-1) ビット右にシフトします。これにより、クイーンの位置 i に対応するビットが右に移動します。
      //1<<(N-k) は、1を (N-k) ビット左にシフトします。これにより、位置 k に対応するビットが左に移動します。
      //両者をビットOR (|) することで、クイーンの位置 i と k に対応するビットが1となり、これが左対角線の攻撃範囲を表します。
      ld=(L>>>(i-1))|(1<<(N-k));
      //クイーンの右対角線上の攻撃範囲を設定する。
      //L>>>(i+1) は、Lを (i+1) ビット右にシフトします。これにより、クイーンの位置 i に対応するビットが右に移動します。
      //1<<(l-1) は、1を (l-1) ビット左にシフトします。これにより、位置 l に対応するビットが左に移動します。
      //両者をビットOR (|) することで、クイーンの位置 i と l に対応するビットが1となり、これが右対角線の攻撃範囲を表します。
      rd=(L>>>(i+1))|(1<<(l-1));
      //クイーンの列の攻撃範囲を設定する。
      //1 は、最初の列（左端）にクイーンがいることを示します。
      //L は、最上位ビットが1であるため、最初の行にクイーンがいることを示します。
      //L>>>i は、Lを i ビット右にシフトし、クイーンの位置 i に対応する列を占有します
      //L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する列を占有します。
      //これらをビットOR (|) することで、クイーンの位置 i と j に対応する列が1となり、これが列の攻撃範囲を表します。
      col=1|L|(L>>>i)|(L>>>j);
      // 最後の列のクイーンj、k、lの対角線を占領しボード上方に移動させる
      //L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する左対角線を占有します。
      //L>>>l は、Lを l ビット右にシフトし、クイーンの位置 l に対応する左対角線を占有します。
      //両者をビットOR (|) することで、クイーンの位置 j と l に対応する左対角線が1となり、これが左対角線の攻撃範囲を表します。
      LD=(L>>>j)|(L>>>l);
      //最後の列の右対角線上の攻撃範囲を設定する。
      //L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する右対角線を占有します。
      //1<<k は、1を k ビット左にシフトし、クイーンの位置 k に対応する右対角線を占有します。
      //両者をビットOR (|) することで、クイーンの位置 j と k に対応する右対角線が1となり、これが右対角線の攻撃範囲を表します。
      RD=(L>>>j)|(1<<k);
      // すべてのサブコンステレーションを数える
      counter=0;
      // すべてのサブコンステレーションを生成する
      setPreQueens(ld,rd,col,k,l,1,j==N-1 ? 3 : 4);
      currentSize=constellations.size();
      // jklとsymとstartはすべてのサブコンステレーションで同じである。
      for (int a=0;a<counter;a++){
        constellations.get(currentSize-a-1)
          .setStartijkl(constellations.get(currentSize-a-1).getStartijkl()|toijkl(i,j,k,l));
      }
    }
  }
  public Main(int sn)
  {
    N=sn;
    presetQueens=4;
    solutions=0;
    N3=N-3;
    N4=N-4;
    L=1<<(N-1);
    L3=1<<N3;
    L4=1<<N4;
  }
  public static void main(String[] args)
  {
    Main main=new Main(14);
    main.genConstellations();
    main.execSolutions();
    main.calcSolutions();
    System.out.println(main.getSolutions());
  }
  //
  int toijkl(int i,int j,int k,int l){ return (i<<15)+(j<<10)+(k<<5)+l;}
  int geti(int ijkl){ return ijkl>>15;}
  int getj(int ijkl){ return (ijkl>>10)&31;}
  int getk(int ijkl){ return (ijkl>>5)&31;}
  int getl(int ijkl){ return ijkl&31;}
  int getjkl(int ijkl){ return ijkl&0b111111111111111;}
  //
  private void SQd0B(int ld,int rd,int col,int row,int free)
  {
    if(row==endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd0B(next_ld,next_rd,next_col,row+1,nextfree);
        }else{
          SQd0B(next_ld,next_rd,next_col,row+1,nextfree);
        }
      }
    }
  }
  private void SQd0BkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd0B((ld|bit)<<2,((rd|bit)>>2)|L3,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1BklB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|L4);
        if(nextfree>0){
          SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1B(int ld,int rd,int col,int row,int free)
  {
    if(row==endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row+1<endmark){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd1B(next_ld,next_rd,next_col,row+1,nextfree);
        }else{
          SQd1B(next_ld,next_rd,next_col,row+1,nextfree);
        }
      }
    }
  }
  private void SQd1BkBlB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1BlB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        int next_ld=((ld|bit)<<2)|1;
        int next_rd=((rd|bit)>>2);
        int next_col=(col|bit);
        nextfree=~(next_ld|next_rd|next_col);
        if(nextfree>0){
          if(row+2<endmark){
            if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
              SQd1B(next_ld,next_rd,next_col,row+2,nextfree);
          }else{
            SQd1B(next_ld,next_rd,next_col,row+2,nextfree);
          }
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1BlkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|L3);
        if(nextfree>0){
          SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1BlBkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd1BkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd1B(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BlkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L3|2);
        if(nextfree>0){
          SQd2B(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BklB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L4|1);
        if(nextfree>0){
          SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BlBkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BkBlB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
        if(nextfree>0){
          SQd2BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BlB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2BkB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd2B(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQd2B(int ld,int rd,int col,int row,int free)
  {
    if(row==endmark){
      if((free&(~1))>0){
        tempcounter++;
      }
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd2B(next_ld,next_rd,next_col,row+1,nextfree);
        }else{
          SQd2B(next_ld,next_rd,next_col,row+1,nextfree);
        }
      }
    }
  }
  // for d>2 but d <small enough>
  private void SQBkBlBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
        if(nextfree>0){
          SQBlBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBlBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==jmark){
      free&=(~1);
      ld|=1;
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
        if(nextfree>0){
          SQB(((ld|bit)<<1),(rd|bit)>>1,col|bit,row+1,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQB(int ld,int rd,int col,int row,int free)
  {
    if(row==endmark){
      tempcounter++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0){
            SQB(next_ld,next_rd,next_col,row+1,nextfree);
          }
        }else{
          SQB(next_ld,next_rd,next_col,row+1,nextfree);
        }
      }
    }
  }
  private void SQBlBkBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBkBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBklBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L4|1);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBlkBjrB(int ld,int rd,int col,int row,int free)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L3|2);
        if(nextfree>0)
          SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree);
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  // for d <big>
  private void SQBjlBkBlBjrB(int ld,int rd,int col,int row,int free)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBkBlBjrB(ld,rd,col,row,free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBjlBlBkBjrB(int ld,int rd,int col,int row,int free)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBlBkBjrB(ld,rd,col,row,free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBjlBklBjrB(int ld,int rd,int col,int row,int free)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBklBjrB(ld,rd,col,row,free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree);
      }
    }
  }
  private void SQBjlBlkBjrB(int ld,int rd,int col,int row,int free)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBlkBjrB(ld,rd,col,row,free);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBlkBjrB( (ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree );
      }
    }
  }
}
