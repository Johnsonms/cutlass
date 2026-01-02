/***************************************************************************************************
 * Simple CuTe Tensor Testing Program
 * Usage: Modify this file to test different layouts, tensors, and print functions
 **************************************************************************************************/

#include <iostream>
#include <cstdio>
#include <vector>

// CuTe includes
#include "example_utils.hpp"
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
      }
      printf("\n");
    }
}
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
int main(int argc, char** argv)
{
  // Create identity tensors that generate coordinates
  // These tensors return their coordinate when accessed
  auto a = make_identity_tensor(make_shape(4, 5));
  print_tensor(a);
  printf("\n");

  // You can also create with specific stride order
  auto b = make_identity_tensor(make_shape(4, 5));
  print_tensor(b);
  printf("\n");
  // Tensor t = make_tensor(counting_iterator<int>(42), make_shape(4,5));
  // for (int i = 0; i < t.size(); ++i)
  //   std::cout << t(i) << " ";
  // std::cout << "\n";
  // print_tensor(t);

  // Construct a TV-layout that maps 8 thread indices and 4 value indices
  //   to 1D coordinates within a 4x8 tensor
  // (T8,V4) -> (M4,N8)
  auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                          Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)
  print_layout(tv_layout);

  // Construct a 4x8 tensor with any layout
  Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
  print_layout(A.layout());
  // Compose A with the tv_layout to transform its shape and order
  Tensor tv = composition(A, tv_layout);                           // (8,4)
  print_layout(tv.layout());
  // Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
  // Tensor  v = tv(threadIdx.x, _);                                  // (4)
  // print_layout(v.layout());

  // Layout s8 = make_layout(Int<8>{});
  // Layout d8 = make_layout(8);

  // Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  // Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

  // Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
  //                             make_stride(Int<12>{},Int<1>{}));
  // Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
  //                               LayoutLeft{});
  // Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
  //                               LayoutRight{});

  // Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
  //                           make_stride(4,make_stride(2,1)));
  // Layout s2xh4_col = make_layout(shape(s2xh4),
  //                               LayoutLeft{});

  // std::cout << "s2xs4 layout:\n";
  // print_layout(s2xs4);
  // print2D(s2xs4);
  // print1D(s2xs4);

  // std::cout << "\ns2xd4_a layout:\n";
  // print_layout(s2xd4_a);
  // print2D(s2xd4_a);
  // print1D(s2xd4_a);

  // std::cout << "\ns2xd4_col layout:\n";
  // print_layout(s2xd4_col);
  // print2D(s2xd4_col);
  // print1D(s2xd4_col);

  // std::cout << "\ns2xd4_row layout:\n";
  // print_layout(s2xd4_row);
  // print2D(s2xd4_row);
  // print1D(s2xd4_row);

  // std::cout << "\ns2xh4 layout:\n";
  // print_layout(s2xh4);
  // print2D(s2xh4);
  // print1D(s2xh4);

  // auto a = Layout<Shape<_4,_2>, Stride<_2,_1>>{};
  // std::cout<<"\nLayout A:\n";
  // std::cout<< "Rank: " << rank(a) << "\n";
  // print_layout(a);
  // print2D(a);
  // print1D(a);

  // std::cout << "Elements: ";
  // for(int i=0; i<size(a); i++){
  //   std::cout<< a(i)<<"\t";
  // }

  // auto b = Layout<Shape <Shape<_4,_2>>, Stride<Stride<_2,_1>>>{};
  // std::cout << "\nLayout B:\n";
  // std::cout << "Rank: " << rank(b) << "\n";

  // auto c = Layout<Shape<_3,Shape<_2,_3>>, Stride<_3,Stride<_12,_1>>>{};
  // print_layout(c);

  // std::cout<<size<1,1>(c) <<"\t"<< size<1,0>(c) << "\t"<< size<0>(c) <<"\n";
  // for(int k=0; k<size<1,1>(c); k++)
  // {
  //   for (int j=0; j<size<1,0>(c); j++)
  //   {
  //     for(int i=0; i<size<0>(c); i++)
  //     {
  //       std::cout<< i <<",("<< j <<","<< k <<") ";
  //     }
  //   }
  // }
  // std::cout<<"\nElements: ";

  // for(int i=0; i<size(c); i++){
  //   std::cout<< c(i) <<" " << idx2crd(i, Shape<_3, _6>{})<<" "<< idx2crd(i, Shape<_3,Shape<_2,_3>>())<<"\t";
  // }

  // A = (6,2):(8,2)
  // B = (4,3):(3,1)
  // auto a = Layout<Shape <_6,_2>,
  //                       Stride<_8,_2>>{};

  // auto b = Layout<Shape <_4,_3>,
  //                       Stride<_3,_1>>{};

  // print_layout(a);
  // printf("\n");
  // print_layout(b);
  // auto a = Layout<Shape <_2,Shape <_1,_6>>,
  //               Stride<_1,Stride<_6,_2>>>{};
  // print_layout(a);
  // auto b = Layout<Shape <_2,Shape <_6, _1>>,
  //               Stride<_1,Stride<_2, _6>>>{};
  // print_layout(b);
  // assert(coalesce(b) == coalesce(a));

  // auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
  // print_layout(result);
  // // Identical to
  // auto same_r = make_layout(coalesce(layout<0>(a)),
  //                         coalesce(layout<1>(a)));
  // print_layout(same_r);
  // std::cout << "\n=== CuTe Tensor Testing Program ===\n\n";

  // // Example 1: Simple 2D Layout
  // std::cout << "Example 1: Simple 2D Layout\n";
  // std::cout << "----------------------------\n";
  // auto layout1 = Layout<Shape <_2, _3>,
  //                      Stride<_1, _2>>{};
  // std::cout << "Layout: "; print(layout1); std::cout << "\n";
  // print_layout(layout1);
  // std::cout << "\n";

  // // Example 2: Hierarchical Layout
  // std::cout << "Example 2: Hierarchical Layout\n";
  // std::cout << "-------------------------------\n";
  // auto layout2 = Layout<Shape <_2,Shape <_1,_6>>,
  //                      Stride<_1,Stride<_6,_2>>>{};
  // std::cout << "Layout: "; print(layout2); std::cout << "\n";
  // print_layout(layout2);
  // std::cout << "\n";

  // // Example 3: Creating a Tensor with Data
  // std::cout << "Example 3: Tensor with Data\n";
  // std::cout << "----------------------------\n";
  // std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  // float* ptr = data.data();

  // auto tensor = make_tensor(ptr,
  //                          make_shape(Int<3>{}, Int<4>{}),
  //                          make_stride(Int<4>{}, Int<1>{}));
  // std::cout << "Tensor: "; print(tensor); std::cout << "\n";
  // std::cout << "Shape:  "; print(shape(tensor)); std::cout << "\n";
  // std::cout << "Stride: "; print(stride(tensor)); std::cout << "\n";
  // std::cout << "Size:   " << size(tensor) << "\n";

  // // Print layout of the tensor
  // print_layout(make_layout(shape(tensor), stride(tensor)));
  // std::cout << "\n";

  // // Example 4: Tensor Slicing
  // std::cout << "Example 4: Tensor Slicing\n";
  // std::cout << "-------------------------\n";
  // auto slice1 = tensor(_, 0);
  // std::cout << "tensor(_, 0): "; print(slice1); std::cout << "\n";
  // std::cout << "Shape:        "; print(shape(slice1)); std::cout << "\n";
  // std::cout << "Stride:       "; print(stride(slice1)); std::cout << "\n\n";

  // // Example 5: More Complex Layout (like GEMM)
  // std::cout << "Example 5: GEMM-like Layout (128x256)\n";
  // std::cout << "--------------------------------------\n";
  // auto gemm_layout = make_layout(make_shape(Int<128>{}, Int<256>{}),
  //                                make_stride(Int<256>{}, Int<1>{}));
  // std::cout << "Layout: "; print(gemm_layout); std::cout << "\n";
  // std::cout << "Shape:  "; print(shape(gemm_layout)); std::cout << "\n";
  // std::cout << "Stride: "; print(stride(gemm_layout)); std::cout << "\n";
  // std::cout << "Size:   " << size(gemm_layout) << "\n\n";

  // // Example 6: Swizzled Layout
  // std::cout << "Example 6: Swizzled Layout\n";
  // std::cout << "--------------------------\n";
  // auto swizzle_layout = composition(Swizzle<3,4,3>{},
  //                                  Layout<Shape<_128, _64>,
  //                                        Stride<  _1, _128>>{});
  // std::cout << "Layout: "; print(swizzle_layout); std::cout << "\n\n";

  // std::cout << "=== End of Testing ===\n\n";

  return 0;
}
