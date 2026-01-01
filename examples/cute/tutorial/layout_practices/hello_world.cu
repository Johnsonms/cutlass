/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

// CuTe Hello World - A simple introduction to CuTe layouts

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
  std::cout << "=== CuTe Hello World ===" << std::endl << std::endl;
  std::cout << "\n";
  auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
  print_latex(layout);
  // 1. Create a simple 1D layout
  std::cout << "1. Simple 1D Layout" << std::endl;
  auto layout_1d = make_layout(Int<8>{});  // 8 elements
  std::cout << "   Layout: " << layout_1d << std::endl;
  std::cout << "   Size: " << size(layout_1d) << std::endl << std::endl;

  // 2. Create a 2D layout (row-major)
  std::cout << "2. 2D Row-Major Layout (4x3)" << std::endl;
  auto layout_2d = make_layout(
      make_shape(Int<4>{}, Int<3>{}),      // 4 rows, 3 columns
      make_stride(Int<3>{}, Int<1>{})      // stride: row=3, col=1
  );
  std::cout << "   Layout: " << layout_2d << std::endl;
  std::cout << "   Element at (0,0): " << layout_2d(0, 0) << std::endl;
  std::cout << "   Element at (1,2): " << layout_2d(1, 2) << std::endl;
  std::cout << "   Total elements: " << size(layout_2d) << std::endl << std::endl;

  // 3. Create a column-major layout
  std::cout << "3. 2D Column-Major Layout (4x3)" << std::endl;
  auto layout_col = make_layout(
      make_shape(Int<4>{}, Int<3>{}),      // 4 rows, 3 columns
      make_stride(Int<1>{}, Int<4>{})      // stride: row=1, col=4
  );
  std::cout << "   Layout: " << layout_col << std::endl;
  std::cout << "   Element at (1,2): " << layout_col(1, 2) << std::endl << std::endl;

  // 4. Print all coordinates and their linear indices
  std::cout << "4. Coordinate Mapping (Row-Major)" << std::endl;
  std::cout << "   ";
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::cout << "(" << i << "," << j << "):" << layout_2d(i, j) << " ";
    }
  }
  std::cout << std::endl << std::endl;

  std::cout << "=== Hello World Complete ===" << std::endl;
  std::cout << "Next: Try modifying the shapes and strides!" << std::endl;

  return 0;
}
