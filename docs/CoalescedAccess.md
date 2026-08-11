## Coalesced Access (合并访问)

## CacheLine

### CacheLine 是warp共享的缓存空间，大小是32 * 4 = 128B，如果32线程访问连续地址，会被合并成一条指令