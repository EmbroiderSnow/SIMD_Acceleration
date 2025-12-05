# SIMD Image Processing Acceleration (基于 SIMD 的图像处理加速)

本项目旨在通过手动编写 **SIMD (Single Instruction, Multiple Data)** 指令集（MMX, SSE2, AVX2）来加速底层的图像处理算法。

项目对比了 **Scalar (标量/串行)** 实现与不同 SIMD 指令集并行实现之间的性能差异，并包含了一套深度性能剖析工具，用于分析程序的瓶颈（计算受限、访存受限或数据重排受限）。

## 🚀 项目功能

本项目主要实现了以下核心图像处理管线，并提供了四种不同的实现版本（Scalar, MMX, SSE2, AVX2）：

1.  **Part 2: 图像淡入淡出 (Fade)**

      * 流程：`YUV420` -\> `ARGB8888` -\> `Alpha Blending` -\> `YUV420`
      * 涉及算法：色彩空间转换、带 Alpha 通道的像素混合。

2.  **Part 3: 图像叠加 (Overlay)**

      * 流程：`YUV420` (两幅图) -\> `RGB888` -\> `Image Overlay` -\> `YUV420`
      * 涉及算法：双路视频流解码、加权叠加、24位 RGB 处理。

3.  **差分性能分析 (Differential Analysis)**

      * 通过构建“阉割版”内核（Only Load/Store, Only Compute, Only Shuffle），量化分析程序的时间开销分布。

## 🛠 支持的指令集

  * **Scalar**: C++ 标准实现（作为基准 Baseline）。
  * **MMX**: 64-bit 寄存器并行（处理 8字节数据）。
  * **SSE2**: 128-bit 寄存器并行（处理 16字节数据）。
  * **AVX2**: 256-bit 寄存器并行（处理 32字节数据）。

> **注意**: 编译配置中已显式禁用了编译器的自动向量化 (`-fno-tree-vectorize`)，以确保测试的是手动编写的 SIMD 代码性能。

## 📂 项目结构

```text
.
├── src
│   ├── core
│   │   ├── Converter.h          # 函数接口声明
│   │   ├── Converter_Scalar.cpp # 标量基准实现
│   │   ├── Converter_MMX.cpp    # MMX 实现
│   │   ├── Converter_SSE.cpp    # SSE2 实现
│   │   └── Converter_AVX.cpp    # AVX2 实现
│   ├── utils
│   │   ├── AlignedMem.h         # 32字节内存对齐分配器
│   │   └── Timer.h              # 高精度计时器
│   └── main.cpp                 # 程序入口与测试管线
├── data                         # 输入数据目录 (需自行放入 .yuv 文件)
├── output                       # 输出结果目录
└── CMakeLists.txt               # 构建配置
```

## 🔨 构建方法

本项目使用 CMake 构建。请确保您的编译器支持 C++17 及相应的 SIMD 指令集（GCC/Clang）。

> 项目没有使用MSVC测试过，CMake配置不保证MSVC可以正常编译。

```bash
mkdir build
cd build
cmake ..
make -j4
```

> **输入数据准备**: 请在项目根目录下创建 `data` 文件夹，并放入 `dem1.yuv` 和 `dem2.yuv` (1920x1080 YUV420格式) 用于测试。

## 🏃 运行与使用

程序通过命令行参数控制运行模式：

```bash
./lab_simd <mode>
```

### 可用模式 (`<mode>`)

1.  **运行基准测试**:

      * `scalar`: 运行标量版本（基准）。
      * `mmx`: 运行 MMX 版本。
      * `sse`: 运行 SSE2 版本。
      * `avx`: 运行 AVX2 版本。

    *示例输出*:

    ```text
    [AVX2] Part 2 (Fade)... 120.5 ms (Avg: 6.0 ms/frame)
    [AVX2] Part 3 (Overlay)... 98.2 ms (Avg: 4.9 ms/frame)
    ```

2.  **运行深度分析**:

      * `analyze`: 运行差分性能分析模式。

## 📊 性能分析指标 (Analysis Mode)

当使用 `analyze` 模式时，程序采用 **差分测试法 (Differential Benchmarking)** 来拆解性能瓶颈。针对 `YUV2RGB` 等核心函数，程序会运行以下变体并统计指标：

### 1\. 统计指标

  * **Baseline Time ($T_{total}$)**: 完整逻辑的运行时间。
  * **MemOnly Time ($T_{mem}$)**: 仅执行内存加载 (Load) 和存储 (Store) 的时间。模拟纯粹的内存带宽压力。
  * **ComputeOnly Time ($T_{comp}$)**: 执行加载和所有算术运算，但不写回内存的时间。用于衡量 ALU (算术逻辑单元) 的压力。
  * **ShuffleOnly Time ($T_{shuffle}$)**: 执行加载、数据重排 (Unpack/Permute/Shuffle) 和存储，但跳过算术运算的时间。

### 2\. 瓶颈判断逻辑

程序会自动计算各部分占比并给出诊断结论：

  * **[MEMORY BOUND]**: 如果 $T_{mem}$ 接近 $T_{total}$，说明瓶颈在于内存带宽，优化 SIMD 计算指令收益甚微。
  * **[COMPUTE BOUND]**: 如果 $T_{comp}$ 接近 $T_{total}$，说明瓶颈在于计算量，适合通过更高级的 SIMD 指令（如 FMA）优化。
  * **[SHUFFLE BOUND]**: 如果 $T_{shuffle}$ 占比较高，说明大量时间消耗在数据格式调整（如 Planar 转 Packed、RGB 结构调整）上。
  * **[LATENCY HIDDEN]**: 如果 $T_{mem} + T_{comp} > T_{total}$，说明 CPU 的乱序执行和流水线技术成功掩盖了部分延迟。

### 运行示例

```text
Running Analysis for: AVX2 (YUV2RGB_RGB888)
------------------------------------------------------------
Baseline:    45.2 ms
MemOnly:     12.1 ms (26.7%)
ComputeOnly: 38.5 ms (85.1%)
ShuffleOnly: 15.3 ms (33.8%) -> Pure Data Layout Overhead

Est. Math Cost: 23.2 ms
Conclusion: COMPUTE BOUND
```

-----

## 📝 备注

  * **内存对齐**: 项目使用了 `AlignedAllocator` 保证内存按 32 字节对齐，以满足 AVX2 `_mm256_load_si256` 等指令的要求。
  * **正确性验证**: 每次运行都会在 `output/` 目录下生成对应的 `.yuv` 文件，可使用 YUV 播放器（如 YUV Player 或 ffplay）验证图像结果是否正确。