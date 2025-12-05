cmake_minimum_required(VERSION 3.10)

project(Lab4_SIMD_Acceleration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# 1. 搜集源码
# -----------------------------
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(lab_simd ${SOURCES})
target_include_directories(lab_simd PRIVATE src)

# -----------------------------
# 2. 全局关闭自动向量化，但保留 SIMD ISA
# -----------------------------
if(MSVC)
    target_compile_options(lab_simd PRIVATE /O2 /arch:AVX2)
else()
    # -mavx2: 允许使用 AVX2 intrinsic
    # -mfma : 允许使用 FMA intrinsic
    #
    # -fno-tree-vectorize: 彻底关闭编译器自动向量化 (包括 loop + SLP)
    #
    # -O3: 允许普通优化，不影响 intrinsic
    target_compile_options(lab_simd PRIVATE 
        -mavx2 
        -mfma
        -O3
        -fno-tree-vectorize
    )
endif()

# -----------------------------
# 3. Scalar 单独降低优化 (-O2)
#    但保持已经禁用自动向量化
# -----------------------------
set_source_files_properties(
    src/core/Converter_Scalar.cpp
    PROPERTIES
    COMPILE_FLAGS "-O2 -fno-tree-vectorize"
)

message(STATUS "Auto-vectorization DISABLED globally.")
message(STATUS "Scalar.cpp uses pure scalar optimization (no SIMD).")
