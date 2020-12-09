// gcc BIT08.c && ./a.out ;
#include <stdio.h>
#include <string.h>

int step;
char pause[32]; 
int fault=0;
int aBoard[8];  //表示用配列
int	TOPBIT,ENDBIT,MASK,SIDEMASK,LASTMASK;
int COUNT2,COUNT4,COUNT8;
int SIZE,SIZEE,TOTAL,UNIQUE;
// 
//
//１０進数を２進数に変換
void con(char* c,int decimal){
  int _decimal=decimal;
  int binary=0;
  int base=1;
  while(decimal>0){
    binary=binary+(decimal%2)*base;
    decimal=decimal/2;
    base=base*10;
  }
  printf("%s 10進数:\t%d\t2進数:\t%05d\n",c,_decimal,binary);
}
//
//ボード表示用
void Display(int y,int BOUND1,int BOUND2,int MODE,int LINE,const char* FUNC,int C,int bm,int left,int down,int right,int mask,int flg_step){
  //MODE=1 TOPBIT,ENDBITを優先する
  //MODE=0 Qを優先する
  //MODE=-1 最下段枝狩りで枝狩りされる場合
  if(fault>y||(y !=0&&fault==y&&MODE==0)){ printf("fault:%d:y:%d####枝狩り####\n\n",fault,y); }
  // printf("Line:%d,Func:%s,Count:%d:Step.%d\n",LINE,FUNC,C,++step);
  // if(BOUND2 !=0){
  //   con("SIDEMASK",SIDEMASK);
  //   con("LASTMASK",LASTMASK);
  //   con("TOPBIT",TOPBIT);
  //   con("ENDBIT",ENDBIT);
  // }
  // printf("\nN=%d no.%d BOUND1:%d:BOUND2:%d:y:%d\n", SIZE, ++count,BOUND1,BOUND2,y);
  con("LEFT",MASK&left);
  con("DOWN",down);
  con("RIGHT",right);
  con("BM",bm);
  int row_cnt=0;
  for (int row=0; row<SIZE; row++) {
    if(row==0){ printf("   ");
      for(int col=0;col<SIZE;col++){ printf("%d ",col); }
      printf("\n");
    }
    if(row==y){ printf(">%d ",row); }
    else{ printf(" %d ",row); }
    int bitmap = aBoard[row];
    int cnt=SIZEE;
    char* s;
    for (int bit=1<<(SIZEE); bit; bit>>=1){
      if(row_cnt>y){ s="-"; }
      else{ s=(bitmap & bit)? "Q": "-"; }
      //MASKの処理
      if(row_cnt==y+1){ if(!(bit&bm)){ s="x"; }
        if(MODE !=-2){ if(aBoard[(row-1)] & bit){ s="D"; }
          if(((aBoard[(row-1)] <<1)& bit)){ s="L"; }
          if(((aBoard[(row-1)] >>1)& bit)){ s="R"; }
        }
      }
      //backtrack1の時の枝狩り

      if(MODE==2){
        if(mask&bit&&row<=BOUND1){
          s="2";
        }
      }
      //SIDEMASKの処理
      //上段枝刈
      if(MODE==3){
        if(mask&bit&&row_cnt==y+1){
          s="S";
        }
      }
      //下段枝刈
      if(MODE==4){
        if(mask&bit&&row_cnt>y+1){
          s="S";
        }
      }

      //最下段枝刈
      if(MODE==5){
        if(mask&bit&&row_cnt==y+1){
         s="M";
       }
      }
      if(row==SIZEE&&BOUND2!=0){
      //ENDBITの処理
        if(ENDBIT&bit&&MODE==6){ s="E"; }
      }
      //TOPBITの処理
      if(row==BOUND1&&cnt==SIZEE&&BOUND2!=0&&MODE==6){ s="T"; }
      cnt--;
      printf("%s ", s);
    }
    printf("\n");
    row_cnt++;
  }
  if(C>0){ fault=0; printf("####処理完了####\n"); }
  else{ if(fault<y){ printf("\n"); } fault=y; }
  if(flg_step){ step++; }
  if(y==SIZE-1){
    printf("N=%d Step.%d %s(),+%d,\n\n",SIZE,step,FUNC,LINE);
  }
  if(strcmp(pause, ".") != 10){ fgets(pause,sizeof(pause),stdin); }
}
/**********************************************/
/* ユニーク解の判定とユニーク解の種類の判定   */
/**********************************************/

typedef unsigned long long uint64;

uint64 reflect_vert (uint64 value)
{
    value = ((value & 0xFFFFFFFF00000000ull) >> 32) | ((value & 0x00000000FFFFFFFFull) << 32);
    value = ((value & 0xFFFF0000FFFF0000ull) >> 16) | ((value & 0x0000FFFF0000FFFFull) << 16);
    value = ((value & 0xFF00FF00FF00FF00ull) >>  8) | ((value & 0x00FF00FF00FF00FFull) <<  8);
    return value;
}

uint64 reflect_horiz (uint64 value)
{
    value = ((value & 0xF0F0F0F0F0F0F0F0ull) >> 4) | ((value & 0x0F0F0F0F0F0F0F0Full) << 4);
    value = ((value & 0xCCCCCCCCCCCCCCCCull) >> 2) | ((value & 0x3333333333333333ull) << 2);
    value = ((value & 0xAAAAAAAAAAAAAAAAull) >> 1) | ((value & 0x5555555555555555ull) << 1);
    return value;
}

uint64 reflect_diag (uint64 value)
{
    uint64 new_value = value & 0x8040201008040201ull; // stationary bits
    new_value |= (value & 0x0100000000000000ull) >> 49;
    new_value |= (value & 0x0201000000000000ull) >> 42;
    new_value |= (value & 0x0402010000000000ull) >> 35;
    new_value |= (value & 0x0804020100000000ull) >> 28;
    new_value |= (value & 0x1008040201000000ull) >> 21;
    new_value |= (value & 0x2010080402010000ull) >> 14;
    new_value |= (value & 0x4020100804020100ull) >>  7;
    new_value |= (value & 0x0080402010080402ull) <<  7;
    new_value |= (value & 0x0000804020100804ull) << 14;
    new_value |= (value & 0x0000008040201008ull) << 21;
    new_value |= (value & 0x0000000080402010ull) << 28;
    new_value |= (value & 0x0000000000804020ull) << 35;
    new_value |= (value & 0x0000000000008040ull) << 42;
    new_value |= (value & 0x0000000000000080ull) << 49;
    return new_value;
}

uint64 rotate_90 (uint64 value)
{
    return reflect_diag (reflect_vert (value));
}

uint64 rotate_180 (uint64 value)
{
    return reflect_horiz (reflect_vert (value));
}

uint64 rotate_270 (uint64 value)
{
    return reflect_diag (reflect_horiz (value));
}
void Check(int BOUND1,int BOUND2) {
  //aBoardを64桁の数字にする
  uint64 r=0;
  int b=SIZE-1;
  for(int col=0;col<SIZE;col++){
    r+=(uint64)aBoard[col]<<b*SIZE; 
    b--;  
  }
  uint64 r_90=rotate_90(r);
  uint64 r_180=rotate_180(r);
  uint64 r_270=rotate_270(r);
  //90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルから180度回転)
  //させても、さらに90度回転(オリジナルから270度回転)させてもオリジナルと同型になる。
  //左右反転させたパターンを加えて２個しかありません。
  if(aBoard[BOUND2]==1){
    if(r==r_90){
      Display(SIZEE, 
          BOUND1,
          BOUND2,
          0,            //MODE
          __LINE__,
          __func__,
          2,            //C
          0,            //bm
          0,0,0,        //left,down,right
          0,            //mask
          1             //stepをカウントするべきか 1: カウントする
          );
      COUNT2++;
      return;
    }
  }
  /*90度回転が同型でなくても180度回転が同型であることもある*/
  //180度回転させて同型になる場合は４個(左右反転×縦横回転)
  if(aBoard[SIZEE]==ENDBIT){
    if(r==r_180){
      COUNT4++;
      Display(SIZEE, 
          BOUND1,
          BOUND2,
          0,            //MODE
          __LINE__,
          __func__,
          4,            //C
          0,            //bm
          0,0,0,        //left,down,right
          0,            //mask
          1             //stepをカウントするべきか 1: カウントする
          );
      return;
    }
  }
  //90度回転,180度回転,270度回転したものと数値を比較して一番小さくなければ枝狩り
  if(r>r_90 || r>r_180||r>r_270){
    return;
  }
  COUNT8++;
  // Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,4,0,0,0,0,0,1,1,1); //表示用
  Display(SIZEE, 
      BOUND1,
      BOUND2,
      0,            //MODE
      __LINE__,
      __func__,
      8,            //C
      0,            //bm
      0,0,0,        //left,down,right
      0,            //mask
      1             //stepをカウントするべきか 1: カウントする
      );
  // con("aBoard180",*aBoard);
}
void Check_o(int BOUND1,int BOUND2) {
  int *own,*you,bit,ptn;
   uint64 r=0;
   int b=SIZE-1;
   for(int col=0;col<SIZE;col++){
    r+=(uint64)aBoard[col]<<b*SIZE; 
    b--;  
   }
  /*90度回転*/
  printf("90_Check::BOUND1:%d:BOUND2:%d:ENDBIT:%d\n",BOUND1,BOUND2,ENDBIT);
  if(aBoard[BOUND2]==1){//aBoard[BOUND2]==1だと現在のクイーンを90度右回転させた時のクイーンの位置
    printf("90_aBoard[BOUND2]:%d:a:%d:b:%d\n",aBoard[BOUND2],*aBoard,*(aBoard+1));
    ptn=2;//２行目が90度回転したらクイーンの位置は右から2番目にくる
    own=aBoard+1;//aBoard+1は２行目aBoard[1]のクイーンの位置
    //２行目が90度回転したら右から2番目のクイーンの位置は2行目のクイーンの位置を右から数えた数分下から数えることになる
    //例えば2行目で右から3番目にクイーンがあれば下から3行目のクイーンの位置が右から2番目だと90度回転して同じということになる
    while(own<=&aBoard[SIZEE]){//２行目から最終行まで90度回転させてチェック
      printf("90_own:%d:aBoard[SIZEE]:%d\n",*own,aBoard[SIZEE]);
      bit=1;
      you=&aBoard[SIZEE];//aBoard[SIZEE]は最終行のクイーンの位置
      printf("90_you:%d:ptn:%d:own:%d:bit:%d\n",*you,ptn,*own,bit);
      while(*you!=ptn&&*own>=bit){//ptnの位置にクイーンがある行をチェックする
        //own ２行目のクイーンの位置が例えば右から5個目だったら bitは 1,2,4,8,16まで回る可能性がある 90度回転させた時に２行目のクイーンは右から5個目だったら下から5個目まで
        printf("90_you:%d!=ptn%d&&own:%d>=bit:%d\n",*you,ptn,*own,bit);
        bit<<=1;
        you--;//最終行のクイーンの位置から一つずつ上の行に移動していく
      }
      printf("90_own:%d:bit:%d\n",*own,bit);
      if(*own>bit){
        printf("90_return\n");
        
        return;
        
       }//例:own aBoard[1]=128 aBoard[8]=2 で抜けた bitは2 
      if(*own<bit)break;//例:own aBoard[1]=16 aBoard[5]まで上がったけどクイーンが右から2になるものはなかったので抜けた
      //own=bit(例えばownが右から4つ目だったら bitが下から4つ目) だと次のループに移動して次の行の90度回転チェック
      own++;
      ptn<<=1;
    }
    /*90度回転して同型なら180度回転も270度回転も同型である*/
    printf("90_own:%d:c2aBpard[SIZEE]%d\n",*own,aBoard[SIZEE]);
    //if(own>&aBoard[SIZEE]){
    if(own>&aBoard[SIZEE]){//ownのループが最終行まで到達（２行目から最終行まで90度回転させてマッチしていたら 90度回転しても同型ということになる
      printf("COUNT2++:%llu",r);
      COUNT2++;
      // Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,2,0,0,0,0,0,1,1,1); //表示用
      Display(SIZEE,
          BOUND1,
          BOUND2,
          0,            //MODE
          __LINE__,
          __func__,
          2,            //C
          0,            //bm
          0,0,0,        //left,down,right
          0,            //mask
          1             //stepをカウントするべきか 1: カウントする
          );
      // con("aBoard90",*aBoard);
      return;
    }
  }
  /*180度回転*/
  //ENDBIT 最終行のクイーンがENDBITの位置にあるときは180度回転して同じ可能性あり
  //１行目の右から3番目にクイーンがあると、ENDBITは左から3番目
  if(aBoard[SIZEE]==ENDBIT){
    printf("180_aBoard[SIZEE]:%d:ENDBIT:%d\n",aBoard[SIZEE],ENDBIT);
    you=&aBoard[SIZEE]-1;//最終行から１行前のクイーンのいち
    own=aBoard+1;//２行目のクイーンの位置
    printf("180_you:%d:own:%d\n",*you,*own);
    //2行目から最終行まで180度回転させて一致するか見ていく
    //例えば2行目のクイーンの位置が右から3番目だったら最終行から1行前のクイーンの位置は左から3番目にあれば180度回転させて一致
    while(own<=&aBoard[SIZEE]){
      printf("180_own:%d:aBoard[SIZEE]:%d\n",*own,aBoard[SIZEE]);
      bit=1;
      ptn=TOPBIT;//ptnを左端から1ビットずつ右に移動していってyouのクイーンの位置がown個分かチェックする
      //例えば、n=12 でownが32だったら ptnを2048,1024,512,256,128,64
      //                               bit  1,2,4,8,16,32
      printf("180_you:%d:ptn:%d:own:%d:bit:%d\n",*you,ptn,*own,bit);
      while(ptn!=*you&&*own>=bit){
        printf("180_bit:%d:you:%d\n",bit,*you);
        bit<<=1;
        ptn>>=1;
      }
      printf("180_own:%d:bit:%d\n",*own,bit);
      if(*own>bit){
        printf("180_return\n");
        return;
      }  
      if(*own<bit)break;
      own++;
      you--;
    //own=bit なら次の行を見る
    }
    /*90度回転が同型でなくても180度回転が同型であることもある*/
    //ownのループが最終行まで到達（２行目から最終行まで180度回転させてマッチしていたら 180度回転しても同型ということになる/
    if(own>&aBoard[SIZEE]){
      printf("180:COUNT4++:%llu:ro:%llu",r,rotate_180(r));
      COUNT4++;
      // Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,4,0,0,0,0,0,1,1,1); //表示用
      Display(SIZEE, 
          BOUND1,
          BOUND2,
          0,            //MODE
          __LINE__,
          __func__,
          4,            //C
          0,            //bm
          0,0,0,        //left,down,right
          0,            //mask
          1             //stepをカウントするべきか 1: カウントする
          );
      // con("aBoard180",*aBoard);
      return;
    }
  }
  /*270度回転*/
  //if(*BOARD1==TOPBIT){
  //270度回転も2行目から順番に回転して同じか比較していくが目的は最小解かどうかのチェックだけ
  if(aBoard[BOUND1]==TOPBIT){
    printf("270_aBoard[BOUND1]:%d:TOPBIT:%d\n",aBoard[BOUND1],TOPBIT);
    ptn=TOPBIT>>1;
    own=aBoard+1;
    printf("270_ptn:%d:own:%d\n",ptn,*own);
    while(own<=&aBoard[SIZEE]){
      printf("270_own:%d:aBoard[SIZEE]:%d\n",*own,aBoard[SIZEE]);
      bit=1;
      you=aBoard;
      printf("270_you:%d:ptn:%d:own:%d:bit:%d\n",*you,ptn,*own,bit);
      while(*you!=ptn&&*own>=bit){
        printf("270_bit:%d:you:%d\n",bit,*you);
        bit<<=1;
        you++;
      }
      printf("270_own:%d:bit:%d\n",*own,bit);
      if(*own>bit){
        printf("270_return\n");
        return;
      }
      if(*own<bit){
        break;
      }
      own++;
      ptn>>=1;
    }
  }
  printf("COUNT8++:%llu",r);
  COUNT8++;
  // Display(SIZEE,BOUND1,BOUND2,0,__LINE__,__func__,8,0,0,0,0,0,1,1,1); //表示用
  Display(SIZEE,    //y
      BOUND1,
      BOUND2,
      0,            //MODE
      __LINE__,
      __func__,
      8,            //C
      0,            //bm
      0,0,0,        //left,down,right
      0,            //mask
      1             //stepをカウントするべきか 1: カウントする
      );
}
/**********************************************/
/* 最上段行のクイーンが角以外にある場合の探索 */
/**********************************************/
void Backtrack2(int y,int left,int down,int right,int BOUND1,int BOUND2){
  int bitmap,bit;
  int flg_s=0;
  int flg_sg=0;
  int flg_m=0;
  /**
    left,down,rightだけだと桁数が盤面より多くなる
    （特にleft)ことがあるのでそれを防ぐために使っている 
    */
  bitmap=MASK&~(left|down|right);
  /**
    BOUND1 1
    BOUND2 6
    TOPBIT   10000000
    SIDEMASK 10000001 BOUND1が1の時は上部サイド枝狩りはない
    LASTMASK 10000001 最終行のクイーンを置けない場所
    ENDBIT   01000000
    0 1 2 3 4 5 6 7
    0 X - - - - - Q X  aBoard[0]=bit=1<<BOUND1(1) 00000010
    1 C - - - - - - -  C:aBoard[BOUND1(1)]==TOPBIT 	aBoard[1]=10000000 270度回転
    2 - - - - - - - - 
    3 - - - - - - - - 
    4 - - - - - - - - 
    5 - - - - - - - - 
    6 - - - - - - - A  A:aBoard[BOUND2(6)]==1 			aBoard[6]=00000001  90度回転
    7 X B - - - - - X  B:aBoard[SIZEE(7)]==ENDBIT 	aBoard[7]=01000000 180度回転

    BOUND1 2
    BOUND2 5
    TOPBIT   10000000
    SIDEMASK 10000001 y=1,6の時に両サイド枝狩り
    LASTMASK 11000011 最終行のクイーンを置けない場所
    ENDBIT   00100000 
    0 1 2 3 4 5 6 7
    0 X X - - - Q X X  aBoard[0]=bit=1<<BOUND1(2) 00000100
    1 X - - - - - - X  
    2 C - - - - - - -  C:aBoard[BOUND1(2)]==TOPBIT 	aBoard[2]=10000000 270度回転
    3 - - - - - - - - 
    4 - - - - - - - - 
    5 - - - - - - - A  A:aBoard[BOUND2(5)]==1 			aBoard[5]=00000001  90度回転
    6 X - - - - - - X  
    7 X X B - - - X X  B:aBoard[SIZEE(7)]==ENDBIT 	aBoard[7]=00100000 180度回転

    BOUND1 3
    BOUND2 4
    TOPBIT   10000000
    SIDEMASK 10000001 y=1,2,5,6の時に両サイド枝狩り
    LASTMASK 11100111 最終行のクイーンを置けない場所
    ENDBIT   00010000 
    0 1 2 3 4 5 6 7
    0 X X X - Q X X X  aBoard[0]=bit=1<<BOUND1(3) 00001000
    1 X - - - - - - X  
    2 X - - - - - - X  
    3 C - - - - - - -  aBoard[BOUND1]==TOPBIT 			aBoard[3]=10000000 270度回転
    4 - - - - - - - A  aBoard[BOUND2]==1 						aBoard[4]=00000001  90度回転
    5 X - - - - - - X  
    6 X - - - - - - X  
    7 X X X B - X X X  aBoard[SIZEE]==ENDBIT 				aBoard[7]=00010000 180度回転
    */
  if(y==SIZEE){
    if(bitmap){
        //最下段枝刈
        // Display(y-1,BOUND1,BOUND2,4,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit),0,flg_s,flg_sg,flg_m); //表示用
        Display(y-1,
            BOUND1,
            BOUND2,
            5,            //MODE
            __LINE__,
            __func__,
            0,            //C
            MASK&~((left)|(down)|(right)), //bm
            (left),(down),(right), //left,down,right
            LASTMASK,            //mask
            0             //stepをカウントするべきか 1: カウントする
            );
      if(!(bitmap&LASTMASK)){/*最下段枝刈り*/
        flg_m=1;
        aBoard[y]=bitmap;
        Check(BOUND1,BOUND2);
      }
    }
  }else{
    if(y<BOUND1){/*上部サイド枝刈り*/
      bitmap|=SIDEMASK;
      bitmap^=SIDEMASK;
      int s=SIDEMASK;
      flg_s=1;
      // Display(y-1,BOUND1,BOUND2,2,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit),0,flg_s,flg_sg,flg_m); //表示用
      Display(y-1,
          BOUND1,
          BOUND2,
          3,            //MODE
          __LINE__,
          __func__,
          0,            //C
          MASK&~((left)|(down)|(right)),//bm
          (left),(down),(right),
          SIDEMASK,            //mask
          0             //stepをカウントするべきか 1: カウントする
          );
    }else if(y==BOUND2){/*下部サイド枝刈り*/
      if(!(down&SIDEMASK)){
        // Display(y,BOUND1,BOUND2,-1,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1,0,flg_s,flg_sg,flg_m); //表示用
        Display(y-1,
            BOUND1,
            BOUND2,
            4,           //MODE
            __LINE__,
            __func__,
            0,            //C
            MASK&~((left)|(down)|(right)), //bm
            (left),(down),(right),
            SIDEMASK,            //mask
            0             //stepをカウントするべきか 1: カウントする
            );
        return;
      }
      if((down&SIDEMASK)!=SIDEMASK)bitmap&=SIDEMASK;
      flg_sg=1;
      // Display(y-1,BOUND1,BOUND2,3,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit),0,flg_s,flg_sg,flg_m); //表示用
      Display(y-1,
          BOUND1,
          BOUND2,
          4,            //MODE
          __LINE__,
          __func__,
          0,            //C
          MASK&~((left)|(down)|(right)), //bm
          (left),(down),(right),
          SIDEMASK,            //mask
          0             //stepをカウントするべきか 1: カウントする
          );
      /**
        兄　ここの枝刈りの遷移をおねがい
        */
    }
    while(bitmap){
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      // Display(y,BOUND1,BOUND2,0,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1,0,flg_s,flg_sg,flg_m); //表示用
      Display(y,
          BOUND1,
          BOUND2,
          0,            //MODE
          __LINE__,
          __func__,
          0,            //C
          MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),//bm
          (left|bit)<<1,(down|bit),(right|bit)>>1,
          0,            //mask
          1             //stepをカウントするべきか 1: カウントする
          );
      Backtrack2(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1,BOUND2);
    }
  }
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
void Backtrack1(int y,int left,int down,int right,int BOUND1){
  int  bitmap, bit;
  int flg_2;
  bitmap=MASK&~(left|down|right);
  if(y==SIZEE){
    if(bitmap){
      aBoard[y]=bitmap;
      /**
        最終行にクイーンを配置する
        */
      COUNT8++;
      // Display(SIZEE,BOUND1,0,1,__LINE__,__func__,8,0,0,0,0,flg_2,0,0,0);  //表示用
      Display(SIZEE,    //y
          BOUND1,
          0,            //BOUND2
          1,            //MODE
          __LINE__,
          __func__,
          8,            //C
          0,            //bm
          0,0,0,        //left,down,right
          0,        //mask
          1             //stepをカウントするべきか 1: カウントする
          );
    }
  }else{
    //y=2の時はこの枝狩りは不要。最適化できないか検討する
    if(y<BOUND1){/*枝刈り : 斜軸反転解の排除*/
      bitmap|=2;
      bitmap^=2;
      flg_2=1;
      //Display(y-1,BOUND1,0,1,__LINE__,__func__,0,MASK&~((left|bit)|(down|bit)|(right|bit)),(left|bit),(down|bit),(right|bit),flg_2,0,0,0); //表示用
      Display(y-1,
          BOUND1,
          0,            //BOUND2
          2,
          __LINE__,
          __func__,
          0,            //C
          MASK&~((left)|(down)|(right)),//bm
          (left),(down),(right),
          2,        //mask
          0             //stepをカウントするべきか 1: カウントする
          );
      /**
        右から２列目を枝狩りする
        ex 10001010->10001000
        一番右端(右上の角)にクイーンがある場合を考えてみます。他の３つの角にクイーン
        を置くことはできないので、ユニーク解であるかどうかを判定するには、右上角から
        左下角を通る斜軸で反転させたパターンとの比較だけになります。突き詰めれば、上
        から２行目のクイーンの位置が右から何番目にあるかと、右から２列目のクイーンの
        位置が上から何番目にあるかを比較するだけで判定することができます。この２つの
        値が同じになることはないからです。
        結局、再帰探索中において y<BOUND1の時に右から２列目に枝狩りを入れておけばユニ
        ーク解になる。  
        右上角から左下角の斜軸で反転から考慮するとy=<BOUND1 になるが
        y=BOUND1の時は利き筋に当たるので枝狩りには入れない
*/
    }
    while(bitmap){
      bitmap^=aBoard[y]=bit=-bitmap&bitmap;
      /**
        bitmap^=aBoard[y]=bit=-bitmap&bitmap;
        クイーンの配置　-bitmap&bitmap で一番右端にクイーンを置く
        bitmap=11110000 -> bitmap=11100000 
        aBoard[y]=00010000
      */
      //Display(y,BOUND1,0,0,__LINE__,__func__,0,MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1),(left|bit)<<1,(down|bit),(right|bit)>>1,flg_2,0,0,0); //表示用
      Display(y,
          BOUND1,
          0,            //BOUND2
          0,            //MODE
          __LINE__,
          __func__,
          0,            //C
          MASK&~((left|bit)<<1|(down|bit)|(right|bit)>>1), //bm
          (left|bit)<<1,(down|bit),(right|bit)>>1,//left,down,right
          0,   //mask
          1             //stepをカウントするべきか 1:カウントする
          );
      Backtrack1(y+1,(left|bit)<<1,down|bit,(right|bit)>>1,BOUND1);
    }
  }
}
//
void NQueens(void) {
  int  bit;
  COUNT8=COUNT4=COUNT2=0;
  SIZEE=SIZE-1;
  MASK=(1<<SIZE)-1;       //255,11111111
  aBoard[0]=1;						//1,  00000001
  //Display(0,2,0,-1,__LINE__,__func__,0,MASK&~((1)<<1|(1)|(1)>>1),1<<1,(1),1>>1,0,0,0,0); //表示用
  Display(0,        //y
      2,            //BOUND1
      0,            //BOUND2
      -1,           //MODE
      __LINE__,
      __func__,
      0,            //C
      MASK&~((1)<<1|(1)|(1)>>1),//bm
      1<<1,1,1>>1,
      0,            //mask
      1             //stepをカウントするべきか 1:カウントする
      );
  /*0行目:000000001(固定)*/
  /*1行目:011111100(選択)*/
  int BOUND1=2;
  //for(BOUND1=2;BOUND1<SIZEE;BOUND1++){
  while(BOUND1<SIZEE){
    aBoard[1]=bit=1<<BOUND1;
    //Display(1,BOUND1,0,0,__LINE__,__func__,0,MASK&~((bit|1)<<1|(bit|1)|(bit|1)>>1),(bit)<<1,(bit),(bit)>>1,0,0,0,0); //表示用
    Display(1,        //y
        BOUND1,
        0,            //BOUND2
        0,            //MODE
        __LINE__,
        __func__,
        0,            //C
        MASK&~((bit|1)<<1|(bit|1)|(bit|1)>>1), //bm
        bit<<1,bit,bit>>1,
        0,            //mask
        1             //stepをカウントするべきか 1:カウントする
        );
    Backtrack1(2,(2|bit)<<1,1|bit,bit>>1,BOUND1);
    BOUND1++;
  }
  TOPBIT=1<<SIZEE;				//128,10000000
  SIDEMASK=LASTMASK=TOPBIT|1; //TOPBIT|1 129: 10000001
  ENDBIT=TOPBIT>>1;           //TOPBIT>>1 64: 01000000
  BOUND1=1;
  int BOUND2=SIZE-2;;
  while(BOUND1<BOUND2){
    //盤面をクリアにする
    /*0行目:000001110(選択)*/
    aBoard[0]=bit=1<<BOUND1;
    //Display(0,BOUND1,BOUND2,0,__LINE__,__func__,0,MASK&~(bit<<1|bit|bit>>1),bit<<1,bit,bit>>1,0,0,0,0); //表示用
    Display(0,        //y
        BOUND1,
        BOUND2,
        0,            //MODE
        __LINE__,
        __func__,
        0,            //C
        MASK&~(bit<<1|bit|bit>>1), //bm
        bit<<1,bit,bit>>1, //left,down,right
        0,            //mask
        1             //stepをカウントするべきか 1:カウントする
        );
    Backtrack2(1,bit<<1,bit,bit>>1,BOUND1,BOUND2);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT>>=1;
    BOUND1++;
    BOUND2--;
  }
  UNIQUE=COUNT8+COUNT4+COUNT2;
  TOTAL=COUNT8*8+COUNT4*4+COUNT2*2;
}
int main(){
  SIZE=8;
  NQueens();
  printf("%2d:%16d%16d\n", SIZE, TOTAL, UNIQUE);
  return 0;
}
