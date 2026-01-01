#include <iostream>
#include <type_traits>

// 简化版示例：演示 typename decltype(...)::NestedType{} 的用法

// 模拟 CuTe 的编译期常量
template <int N>
struct Constant {
    static constexpr int value = N;
};

// 模拟 TiledMMA 类
template <typename ThreadLayout>
struct TiledMMA {
    // 嵌套类型定义
    using AtomThrID = ThreadLayout;

    ThreadLayout thread_layout;
};

// 模拟 make_tile 函数
template <typename T>
auto make_tile(T const& shape) {
    std::cout << "make_tile called with type: "
              << typeid(T).name() << std::endl;
    return shape;
}

int main() {
    // 创建一个 TiledMMA 实例
    TiledMMA<Constant<128>> my_mma;

    std::cout << "=== C++ 语法拆解演示 ===\n\n";

    // 步骤 1: decltype 获取类型
    std::cout << "1. decltype(my_mma) 的类型是: ";
    std::cout << typeid(decltype(my_mma)).name() << "\n\n";

    // 步骤 2: 访问嵌套类型 (需要 typename)
    using NestedType = typename decltype(my_mma)::AtomThrID;
    std::cout << "2. 嵌套类型 AtomThrID 是: ";
    std::cout << typeid(NestedType).name() << "\n";
    std::cout << "   值 = " << NestedType::value << "\n\n";

    // 步骤 3: 创建临时对象
    std::cout << "3. 创建临时对象: typename decltype(my_mma)::AtomThrID{}\n\n";

    // 步骤 4: 传递给 make_tile
    std::cout << "4. 调用 make_tile:\n";
    auto result = make_tile(typename decltype(my_mma)::AtomThrID{});

    std::cout << "\n=== 等价写法对比 ===\n\n";

    // 等价写法 1：分步写
    using MyType = typename decltype(my_mma)::AtomThrID;
    MyType temp_obj{};
    auto result1 = make_tile(temp_obj);

    // 等价写法 2：使用 auto
    auto temp_obj2 = typename decltype(my_mma)::AtomThrID{};
    auto result2 = make_tile(temp_obj2);

    // 原始写法：一行搞定
    auto result3 = make_tile(typename decltype(my_mma)::AtomThrID{});

    std::cout << "三种写法都是等价的！\n";

    return 0;
}

/*
编译运行：
g++ -std=c++17 cpp_explanation_example.cpp -o cpp_example
./cpp_example
*/
