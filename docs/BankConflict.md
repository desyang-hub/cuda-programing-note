## Bank Conflict

### 什么是Bank Conflict

Bank是共享内存访问的单元，最大允许32路并发访问，访问bank索引通过地址 (addr / 4) % 32进行计算，如果互不冲突
例如 
```
__shared__ float s[32];
s[threadIdx.x] // 显然这里就是互不冲突的

__shared__ float s1[M][33];

// 如果按列访问, 那么这里计算的实际地址 (threadIdx.x * 33 + 1) % 32 => 1, 2,3,4... 互不冲突
s1[threadIdx.x][1]
