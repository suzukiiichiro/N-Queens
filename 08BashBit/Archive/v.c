#include <stdio.h>
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
int main(){
   int aBoard[8];
   aBoard[7]=1;
   uint64 r=0;
   int b=7;
   for(int col=0;col<8;col++){
    r+=(uint64)aBoard[col]<<b*8; 
    b--;  
   }
   r=292736724579125280;
printf("r:%llu:r:%llu\n",r,rotate_180(r));
if(r==rotate_180(r)){
 printf("equal\n");
}
return 0;
}

