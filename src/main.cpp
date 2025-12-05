#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional> // for std::function
#include <map>
#include <cstring>

#include "core/ImageTypes.h"
#include "core/Converter.h"
#include "utils/Timer.h"

// 全局配置
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int Y_SIZE = WIDTH * HEIGHT;
const int UV_SIZE = Y_SIZE / 4;

// 【修改点1】设置重复运行次数，延长测试时间以获得稳定结果
const int REPEAT_COUNT = 20; 

// ---------------------------------------------------------
// 文件操作辅助函数
// ---------------------------------------------------------
bool readYUV(const std::string& filename, YUVFrame& frame) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    file.read(reinterpret_cast<char*>(frame.Y), Y_SIZE);
    file.read(reinterpret_cast<char*>(frame.U), UV_SIZE);
    file.read(reinterpret_cast<char*>(frame.V), UV_SIZE);
    auto ret = file.gcount() == UV_SIZE;
    file.close();
    return ret; 
}

void appendYUV(std::ofstream& file, const YUVFrame& frame) {
    file.write(reinterpret_cast<const char*>(frame.Y), Y_SIZE);
    file.write(reinterpret_cast<const char*>(frame.U), UV_SIZE);
    file.write(reinterpret_cast<const char*>(frame.V), UV_SIZE);
}

// ---------------------------------------------------------
// Part 2 通用测试管线
// ---------------------------------------------------------
using FuncYUV2ARGB = std::function<void(const YUVFrame&, RGBFrame&, uint8_t)>;
using FuncBlend    = std::function<void(RGBFrame&, uint8_t)>;
using FuncRGB2YUV  = std::function<void(const RGBFrame&, YUVFrame&)>;

double runPart2_Pipeline(const std::string& label, 
                         const std::string& outName,
                         FuncYUV2ARGB funcY2R, 
                         FuncBlend funcBlend, 
                         FuncRGB2YUV funcR2Y) 
{
    std::cout << "[" << label << "] Part 2 (Fade)... ";
    std::flush(std::cout);

    // 1. 准备数据
    YUVFrame input(WIDTH, HEIGHT);
    if (!readYUV("../data/dem1.yuv", input)) {
        std::cerr << "Error: dem1.yuv missing!" << std::endl;
        return 0.0;
    }
    RGBFrame midFrame(WIDTH, HEIGHT, RGBFormat::ARGB8888); 
    YUVFrame output(WIDTH, HEIGHT);
    
    std::string outPath = "../output/" + outName + ".yuv";
    std::ofstream outFile(outPath, std::ios::binary);

    // 预热 (Warm-up): 空跑一次，让 OS 分配物理页，让 CPU 进入高频状态
    funcY2R(input, midFrame, 255);
    funcBlend(midFrame, 128);
    funcR2Y(midFrame, output);

    // 2. 运行并计时
    Timer t;
    int frames = 0;
    int total_ops = 0;

    // 【修改点2】外层增加重复循环
    for (int r = 0; r < REPEAT_COUNT; ++r) {
        for (int alpha = 1; alpha <= 255; alpha += 3) {
            // 核心计算 (每次都跑)
            funcY2R(input, midFrame, 255);       
            funcBlend(midFrame, alpha);          
            funcR2Y(midFrame, output);           
            
            // 【修改点3】IO 优化：只在第一轮写入文件
            // 避免磁盘 IO 成为瓶颈，掩盖了 SIMD 的性能优势
            if (r == 0) {
                appendYUV(outFile, output);
                frames++;
            }
            total_ops++;
        }
    }
    
    double ms = t.elapsed();
    // 输出总耗时，以及单帧平均耗时
    std::cout << ms << " ms (Avg: " << (ms / total_ops) << " ms/frame | Loops: " << REPEAT_COUNT << ")" << std::endl;
    return ms;
}

// ---------------------------------------------------------
// Part 3 通用测试管线
// ---------------------------------------------------------
using FuncYUV2RGB_P3 = std::function<void(const YUVFrame&, RGBFrame&)>;
using FuncOverlay    = std::function<void(const RGBFrame&, const RGBFrame&, RGBFrame&, uint8_t)>;

double runPart3_Pipeline(const std::string& label, 
                         const std::string& outName,
                         FuncYUV2RGB_P3 funcY2R, 
                         FuncOverlay funcOverlay, 
                         FuncRGB2YUV funcR2Y) 
{
    std::cout << "[" << label << "] Part 3 (Overlay)... ";
    std::flush(std::cout);

    YUVFrame src1(WIDTH, HEIGHT);
    YUVFrame src2(WIDTH, HEIGHT);
    RGBFrame rgb1(WIDTH, HEIGHT, RGBFormat::RGB888); 
    RGBFrame rgb2(WIDTH, HEIGHT, RGBFormat::RGB888);
    RGBFrame blended(WIDTH, HEIGHT, RGBFormat::RGB888);
    YUVFrame out(WIDTH, HEIGHT);

    if (!readYUV("../data/dem1.yuv", src1) || !readYUV("../data/dem2.yuv", src2)) {
        std::cerr << "Error: dem1.yuv or dem2.yuv missing!" << std::endl;
        return 0.0;
    }

    std::string outPath = "../output/" + outName + ".yuv";
    std::ofstream outFile(outPath, std::ios::binary);

    // 预热
    funcY2R(src1, rgb1);
    funcY2R(src2, rgb2);
    funcOverlay(rgb1, rgb2, blended, 128);
    funcR2Y(blended, out);

    Timer t;
    int frames = 0;
    int total_ops = 0;

    // 【修改点2】外层增加重复循环
    for (int r = 0; r < REPEAT_COUNT; ++r) {
        for (int alpha = 1; alpha <= 255; alpha += 3) {
            funcY2R(src1, rgb1);
            funcY2R(src2, rgb2);
            funcOverlay(rgb1, rgb2, blended, alpha);
            funcR2Y(blended, out);
            
            // 【修改点3】IO 优化：只在第一轮写入文件
            if (r == 0) {
                appendYUV(outFile, out);
                frames++;
            }
            total_ops++;
        }
    }

    double ms = t.elapsed();
    std::cout << ms << " ms (Avg: " << (ms / total_ops) << " ms/frame | Loops: " << REPEAT_COUNT << ")" << std::endl;
    return ms;
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main(int argc, char* argv[]) {
    // 【修改点4】移除默认 "all" 模式，强制用户指定
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scalar|mmx|sse>" << std::endl;
        std::cerr << "Example: " << argv[0] << " sse" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::cout << "=== Lab 4.1 SIMD Benchmark (" << mode << ") | Repeat: " << REPEAT_COUNT << "x ===" << std::endl;

    // --- 1. Run Scalar ---
    // 使用 else if 互斥执行，防止相互干扰
    if (mode == "scalar") {
        runPart2_Pipeline("Scalar", "part2_scalar",
            Scalar::YUV2RGB_ARGB8888, 
            Scalar::AlphaBlend, 
            Scalar::RGB2YUV_ARGB8888);

        runPart3_Pipeline("Scalar", "part3_scalar",
            Scalar::YUV2RGB_RGB888, 
            Scalar::ImageOverlay, 
            Scalar::RGB2YUV_RGB888);
    }
    // --- 2. Run MMX ---
    else if (mode == "mmx") {
        runPart2_Pipeline("MMX   ", "part2_mmx",
            MMX::YUV2RGB_ARGB8888, 
            MMX::AlphaBlend,        
            MMX::RGB2YUV_ARGB8888    
        );

        runPart3_Pipeline("MMX   ", "part3_mmx",
            MMX::YUV2RGB_RGB888, 
            MMX::ImageOverlay,      
            MMX::RGB2YUV_RGB888      
        );
    }
    // --- 3. Run SSE ---
    else if (mode == "sse") {
        runPart2_Pipeline("SSE2  ", "part2_sse",
            SSE::YUV2RGB_ARGB8888, 
            SSE::AlphaBlend, 
            SSE::RGB2YUV_ARGB8888
        );

        runPart3_Pipeline("SSE2  ", "part3_sse",
            SSE::YUV2RGB_RGB888, 
            SSE::ImageOverlay, 
            SSE::RGB2YUV_RGB888
        );
    }
    else if (mode == "avx") {
        runPart2_Pipeline("AVX2  ", "part2_avx",
            AVX::YUV2RGB_ARGB8888, 
            AVX::AlphaBlend, 
            AVX::RGB2YUV_ARGB8888
        );

        runPart3_Pipeline("AVX2  ", "part3_avx",
            AVX::YUV2RGB_RGB888, 
            AVX::ImageOverlay, 
            AVX::RGB2YUV_RGB888
        );
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}