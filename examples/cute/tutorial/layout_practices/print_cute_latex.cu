#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

int main() {
  // 1) 一个简单的 rank-2 layout：看坐标->index 映射（彩色表）
  auto l0 = make_layout(make_shape(Int<16>{}, Int<16>{}),
                        make_stride(Int<16>{}, Int<1>{}));
  print_latex(l0);

  // 2) 一个更"硬件味"的对象：TiledMMA（更接近 warp/thread/fragment 视角）
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
  using MMA = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  auto tiled_mma = make_tiled_mma(MMA{}, make_layout(make_shape(Int<2>{}, Int<2>{})));
  print_latex(tiled_mma);
  std::cout << "\n";
  A_frag_layout =
    Layout<
      Shape< Shape<Int<8>, Int<2>>,  Shape<Int<4>, Int<4>> >,
      Stride<Stride<Int<16>, Int<128>>, Stride<Int<1>, Int<4>> >
    >{};
  print_latex(A_frag_layout);
#else
  // 如果编译时没有开启 SM80 支持，也可以打印嵌套 layout
  auto nested_layout = make_layout(
    make_shape(make_shape(Int<2>{}, Int<2>{}), make_shape(Int<8>{}, Int<8>{})),
    make_stride(make_stride(Int<1>{}, Int<2>{}), make_stride(Int<16>{}, Int<4>{}))
  );
  print_latex(nested_layout);
#endif

  return 0;
}

