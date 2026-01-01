/***************************************************************************************************
 * 对比 logical_divide 和 local_tile 的使用
 **************************************************************************************************/

#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main()
{
  std::cout << "═══════════════════════════════════════════\n";
  std::cout << "  logical_divide vs local_tile 对比\n";
  std::cout << "═══════════════════════════════════════════\n\n";

  // 创建 8×8 数据
  int data[64];
  for (int i = 0; i < 64; ++i) data[i] = i;
  int* data_ptr = &data[0];

  auto layout = make_layout(make_shape(Int<8>{}, Int<8>{}),
                            make_stride(Int<8>{}, Int<1>{}));
  auto tensor = make_tensor(data_ptr, layout);

  auto tile_shape = make_shape(Int<2>{}, Int<2>{});  // 2×2 的 tile

  std::cout << "原始数据: 8×8 矩阵，值从 0 到 63\n";
  std::cout << "tile 大小: 2×2\n";
  std::cout << "分成 4×4 = 16 个 tiles\n\n";

  // ═══════════════════════════════════════
  // 方法 1: logical_divide
  // ═══════════════════════════════════════
  std::cout << "━━━ 方法1: logical_divide ━━━\n\n";

  auto tiled_layout = logical_divide(layout, tile_shape);
  std::cout << "原始 layout: " << layout << "\n";
  std::cout << "分块后 layout: " << tiled_layout << "\n\n";

  // 创建使用分层 layout 的 tensor
  auto tiled_tensor = make_tensor(data_ptr, tiled_layout);
  std::cout << "tiled_tensor 的 shape: " << shape(tiled_tensor) << "\n";
  std::cout << "解释: ((2,2),(4,4)) = ((tile内), (tile间))\n\n";

  // 访问 tile (1,2) 的元素
  std::cout << "访问 tile(1,2) 的数据（手动索引）:\n";
  std::cout << "  tiled_tensor(0,0,1,2) = " << tiled_tensor(0,0,1,2) << "\n";
  std::cout << "  tiled_tensor(0,1,1,2) = " << tiled_tensor(0,1,1,2) << "\n";
  std::cout << "  tiled_tensor(1,0,1,2) = " << tiled_tensor(1,0,1,2) << "\n";
  std::cout << "  tiled_tensor(1,1,1,2) = " << tiled_tensor(1,1,1,2) << "\n";
  std::cout << "  索引格式: (tile内行, tile内列, tile行号, tile列号)\n\n";

  // ═══════════════════════════════════════
  // 方法 2: local_tile
  // ═══════════════════════════════════════
  std::cout << "━━━ 方法2: local_tile ━━━\n\n";

  auto tile = local_tile(tensor, tile_shape, make_coord(1, 2));
  std::cout << "tile shape: " << shape(tile) << "\n";
  std::cout << "tile 就是一个普通的 2×2 tensor\n\n";

  std::cout << "访问 tile(1,2) 的数据（直接访问）:\n";
  std::cout << "  tile(0,0) = " << tile(0,0) << "\n";
  std::cout << "  tile(0,1) = " << tile(0,1) << "\n";
  std::cout << "  tile(1,0) = " << tile(1,0) << "\n";
  std::cout << "  tile(1,1) = " << tile(1,1) << "\n";
  std::cout << "  索引格式: (行, 列) - 简单直接！\n\n";

  // ═══════════════════════════════════════
  // 对比总结
  // ═══════════════════════════════════════
  std::cout << "━━━ 对比总结 ━━━\n\n";

  std::cout << "logical_divide:\n";
  std::cout << "  优点: 保留完整的分层结构信息\n";
  std::cout << "  用途: 理解布局、手动控制索引、某些高级操作\n";
  std::cout << "  访问: tiled_tensor(tile内坐标, tile间坐标) - 复杂\n\n";

  std::cout << "local_tile:\n";
  std::cout << "  优点: 直接获取特定 tile，简单易用\n";
  std::cout << "  用途: 提取数据块、最常用的方式\n";
  std::cout << "  访问: tile(行, 列) - 简单直接\n\n";

  // ═══════════════════════════════════════
  // 实际应用场景
  // ═══════════════════════════════════════
  std::cout << "━━━ 实际应用场景 ━━━\n\n";

  std::cout << "场景1: GEMM 中提取 CTA 的数据\n";
  std::cout << "  ✅ 使用 local_tile:\n";
  std::cout << "     auto gA = local_tile(mA, mma_tiler, cta_coord);\n";
  std::cout << "     // 直接得到当前 CTA 的数据块\n\n";

  std::cout << "场景2: 遍历所有 tiles\n";
  std::cout << "  ✅ 使用 local_tile:\n";
  std::cout << "     for (int i = 0; i < num_tiles_m; ++i)\n";
  std::cout << "       for (int j = 0; j < num_tiles_n; ++j)\n";
  std::cout << "         auto tile = local_tile(tensor, shape, make_coord(i,j));\n\n";

  std::cout << "场景3: 需要理解分层结构\n";
  std::cout << "  ✅ 使用 logical_divide:\n";
  std::cout << "     auto tiled = logical_divide(layout, tile_shape);\n";
  std::cout << "     print(tiled);  // 查看结构\n\n";

  // ═══════════════════════════════════════
  // 重要提示
  // ═══════════════════════════════════════
  std::cout << "━━━ 重要提示 ━━━\n\n";

  std::cout << "⚠️  不需要先 logical_divide 再 local_tile！\n";
  std::cout << "   它们是两个独立的操作，服务不同目的。\n\n";

  std::cout << "✅ 大多数情况下，直接使用 local_tile 即可！\n";
  std::cout << "   这是最直接、最常用的方式。\n\n";

  std::cout << "📖 在 01_mma_sm100.cu 中:\n";
  std::cout << "   - 第 170 行使用 local_tile 提取 CTA 数据\n";
  std::cout << "   - 没有使用 logical_divide\n";
  std::cout << "   - local_tile 是主要的数据提取方式\n\n";

  return 0;
}
