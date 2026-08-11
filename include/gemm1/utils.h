#pragma once

#define OFFSET(row, col, cols) ((cols) * (row) + col)
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])