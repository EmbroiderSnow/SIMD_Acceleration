#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include "core/ImageTypes.h"
#include "core/Converter.h"
#include "utils/Timer.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;
const int Y_SIZE = WIDTH * HEIGHT;
const int UV_SIZE = Y_SIZE / 4;
const int FRAME_SIZE = Y_SIZE + 2 * UV_SIZE;

bool readYUV(const std::string& filename, YUVFrame& frame) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file " << filename << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(frame.Y), Y_SIZE);
    file.read(reinterpret_cast<char*>(frame.U), UV_SIZE);
    file.read(reinterpret_cast<char*>(frame.V), UV_SIZE);

    if (!file) {
        std::cerr << "Error: File size too small or read error." << std::endl;
        return false;
    }
    file.close();
    return true;
}

void appendYUV(std::ofstream& file, const YUVFrame& frame) {
    file.write(reinterpret_cast<const char*>(frame.Y), Y_SIZE);
    file.write(reinterpret_cast<const char*>(frame.U), UV_SIZE);
    file.write(reinterpret_cast<const char*>(frame.V), UV_SIZE);
}

void runPart2_Scalar() {
    std::cout << "Starting Part 2 (Scalar Implementation)..." << std::endl;

    // 1. 准备数据容器
    // 输入使用 Aligned 内存的 YUVFrame
    YUVFrame inputFrame(WIDTH, HEIGHT);
    // 中间处理用的 ARGBFrame (Part 2 要求 ARGB8888 [cite: 29])
    RGBFrame middleFrame(WIDTH, HEIGHT, RGBFormat::ARGB8888);
    // 输出容器
    YUVFrame outputFrame(WIDTH, HEIGHT);

    // 2. 读取输入文件
    // 注意：请确保 data 目录下有 dem1.yuv
    std::string inputPath = "../data/dem1.yuv";
    std::string outputPath = "../output/part2_scalar_result.yuv";

    if (!readYUV(inputPath, inputFrame)) {
        return;
    }

    // 打开输出文件 (截断模式，重新写入)
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open output file " << outputPath << std::endl;
        return;
    }

    // 3. 处理循环 [cite: 29]
    // Alpha 范围: 1~255, 步长 3
    int frameCount = 0;
    Timer totalTimer; // 记录总计算时间

    std::cout << "Processing frames..." << std::endl;

    for (int alpha = 1; alpha <= 255; alpha += 3) {
        // --- 核心计算区 (开始计时) ---
        // Lab 要求加速这部分: YUV2RGB, AlphaBlend, RGB2YUV 
        
        // 步骤 A: YUV420 -> ARGB8888
        // 先转成不透明的 (alpha=255)，稍后做混合
        Scalar::YUV2RGB_ARGB8888(inputFrame, middleFrame, 255);

        // 步骤 B: Alpha 混合计算
        // 公式: New = A * RGB / 256 [cite: 30]
        // 这里实际上是对图像做亮度衰减 (Fade)
        Scalar::AlphaBlend(middleFrame, static_cast<uint8_t>(alpha));

        // 步骤 C: ARGB8888 -> YUV420
        // 将混合后的图像转回 YUV 存入 outputFrame [cite: 31]
        Scalar::RGB2YUV_ARGB8888(middleFrame, outputFrame);
        
        // --- 核心计算区 (结束计时，写文件不算在内) ---

        // 步骤 D: 写入文件
        appendYUV(outFile, outputFrame);
        frameCount++;
    }

    double elapsed = totalTimer.elapsed();
    std::cout << "Part 2 Done." << std::endl;
    std::cout << "Frames generated: " << frameCount << std::endl; // 应该是 84 幅 [cite: 29]
    std::cout << "Total Calculation Time: " << elapsed << " ms" << std::endl;
    std::cout << "Average Time per Frame: " << elapsed / frameCount << " ms" << std::endl;
    std::cout << "Output saved to: " << outputPath << std::endl;
    outFile.close();
}

// ---------------------------------------------------------
// Part 3: 两幅图像叠加的渐变处理 (Scalar)
// ---------------------------------------------------------
void runPart3_Scalar() {
    std::cout << "Starting Part 3 (Scalar Implementation)..." << std::endl;

    // 1. 准备文件路径
    std::string inputPath1 = "../data/dem1.yuv";
    std::string inputPath2 = "../data/dem2.yuv"; // 需要两幅图
    std::string outputPath = "../output/part3_scalar_result.yuv";

    // 2. 准备数据容器
    // 输入 buffers
    YUVFrame srcFrame1(WIDTH, HEIGHT);
    YUVFrame srcFrame2(WIDTH, HEIGHT);
    
    // 中间 buffers (注意：Part 3 要求使用 RGB888 格式)
    RGBFrame rgbFrame1(WIDTH, HEIGHT, RGBFormat::RGB888);
    RGBFrame rgbFrame2(WIDTH, HEIGHT, RGBFormat::RGB888);
    RGBFrame blendedFrame(WIDTH, HEIGHT, RGBFormat::RGB888);

    // 输出 buffer
    YUVFrame outFrame(WIDTH, HEIGHT);

    // 3. 读取输入文件
    if (!readYUV(inputPath1, srcFrame1)) {
        std::cerr << "Failed to read " << inputPath1 << std::endl;
        return;
    }
    if (!readYUV(inputPath2, srcFrame2)) {
        std::cerr << "Failed to read " << inputPath2 << std::endl;
        return;
    }

    // 打开输出文件
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open output file " << outputPath << std::endl;
        return;
    }

    // 4. 处理循环
    // Alpha 范围: 1~255, 步长 3 [Lab要求]
    int frameCount = 0;
    Timer totalTimer; 

    std::cout << "Processing Part 3 frames (Overlay)..." << std::endl;

    for (int alpha = 1; alpha <= 255; alpha += 3) {
        // --- 核心计算区 (开始计时) ---
        
        // 步骤 A: YUV420 -> RGB888 (两幅图都要转)
        // 注意：这里调用的是 RGB888 的版本，不是 ARGB8888
        Scalar::YUV2RGB_RGB888(srcFrame1, rgbFrame1);
        Scalar::YUV2RGB_RGB888(srcFrame2, rgbFrame2);

        // 步骤 B: 图像叠加 (Overlay)
        // 这里的 alpha 控制 src2 的权重：Result = Src1*(1-a) + Src2*a
        Scalar::ImageOverlay(rgbFrame1, rgbFrame2, blendedFrame, static_cast<uint8_t>(alpha));

        // 步骤 C: RGB888 -> YUV420
        Scalar::RGB2YUV_RGB888(blendedFrame, outFrame);
        
        // --- 核心计算区 (结束计时) ---

        // 步骤 D: 写入文件
        appendYUV(outFile, outFrame);
        frameCount++;
    }

    double elapsed = totalTimer.elapsed();
    std::cout << "Part 3 Done." << std::endl;
    std::cout << "Frames generated: " << frameCount << std::endl;
    std::cout << "Total Calculation Time: " << elapsed << " ms" << std::endl;
    std::cout << "Average Time per Frame: " << elapsed / frameCount << " ms" << std::endl;
    std::cout << "Output saved to: " << outputPath << std::endl;
    outFile.close();
}

int main() {
    // 可以在这里增加简单的参数解析来选择运行 Part2 还是 Part3
    // 目前默认运行 Part 2 Scalar Baseline
    runPart2_Scalar();
    runPart3_Scalar();

    return 0;
}