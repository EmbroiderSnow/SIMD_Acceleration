cmake_minimum_required(VERSION 3.10)

# 项目名称
project(Lab4_SIMD_Acceleration)

# 设置 C++ 标准 (C++17 为了支持 aligned_alloc 和更现代的语法)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# 1. 自动搜集源文件
# --------------------------------------------------------------------------
# 递归寻找 src 目录下所有的 .cpp 文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 创建可执行文件
add_executable(lab_simd ${SOURCES})

# 指定头文件搜索路径 (让你能直接 #include "core/ImageTypes.h")
target_include_directories(lab_simd PRIVATE src)

# --------------------------------------------------------------------------
# 2. 全局编译选项 (针对优化后的代码)
# --------------------------------------------------------------------------
# 我们默认开启所有最高级的指令集支持，以便编写 AVX/SSE 代码
if(MSVC)
    # Windows (Visual Studio) 环境
    target_compile_options(lab_simd PRIVATE /arch:AVX2 /O2)
else()
    # Linux (GCC/Clang) 环境
    # -mavx2: 启用 AVX2 指令集 (同时也隐含了 AVX, SSE4, SSE2, MMX)
    # -mfma:  启用融合乘加指令 (FMA)
    # -O3:    最高级别优化 (循环展开、内联等)
    target_compile_options(lab_simd PRIVATE -mavx2 -mfma -O3)
endif()

# --------------------------------------------------------------------------
# 3. 【核心配置】针对 Scalar 基准代码禁用 SIMD
# --------------------------------------------------------------------------
# 目标文件: src/core/Converter_Scalar.cpp
# 目的: 保证 Baseline 是纯粹的标量代码，作为公正的对比基准。

if(NOT MSVC)
    # Linux/GCC/Clang 配置
    set_source_files_properties(
        src/core/Converter_Scalar.cpp
        PROPERTIES
        # -O2: 使用常规优化，不使用 O3 (O3 有时过于激进)
        # -fno-tree-vectorize: 显式禁止编译器尝试自动向量化循环
        # 注意: 我们不使用 -mno-sse 等标志，因为这可能会导致无法链接标准库或其他问题，
        # -fno-tree-vectorize 是控制编译器行为的标准做法。
        COMPILE_FLAGS "-O2 -fno-tree-vectorize"
    )
else()
    # Windows MSVC 配置 (MSVC 的控制粒度较粗)
    set_source_files_properties(
        src/core/Converter_Scalar.cpp
        PROPERTIES
        # /Ob1: 只内联显式标记的函数
        # /No-vectorize: 虽然 MSVC 没有直接的 no-vectorize flag 对应单个文件，
        # 但通常降级到默认优化即可。这里不做特殊处理，依赖 O2 默认行为。
    )
endif()

# --------------------------------------------------------------------------
# 4. 输出一些构建信息
# --------------------------------------------------------------------------
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Source Files: ${SOURCES}")
message(STATUS "Note: Auto-vectorization is DISABLED for Converter_Scalar.cpp")