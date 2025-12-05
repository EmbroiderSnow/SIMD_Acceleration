#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
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

const int REPEAT_COUNT = 50; 

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

    YUVFrame input(WIDTH, HEIGHT);
    if (!readYUV("../data/dem1.yuv", input)) {
        std::cerr << "Error: dem1.yuv missing!" << std::endl;
        return 0.0;
    }
    RGBFrame midFrame(WIDTH, HEIGHT, RGBFormat::ARGB8888); 
    YUVFrame output(WIDTH, HEIGHT);
    
    std::string outPath = "../output/" + outName + ".yuv";
    std::ofstream outFile(outPath, std::ios::binary);

    funcY2R(input, midFrame, 255);
    funcBlend(midFrame, 128);
    funcR2Y(midFrame, output);

    Timer t;
    int frames = 0;
    int total_ops = 0;

    for (int r = 0; r < REPEAT_COUNT; ++r) {
        for (int alpha = 1; alpha <= 255; alpha += 3) {
            funcY2R(input, midFrame, 255);       
            funcBlend(midFrame, alpha);          
            funcR2Y(midFrame, output);           
            
            if (r == 0) {
                appendYUV(outFile, output);
                frames++;
            }
            total_ops++;
        }
    }
    
    double ms = t.elapsed();
    std::cout << ms << " ms (Avg: " << (ms / total_ops) << " ms/frame | Loops: " << REPEAT_COUNT << ")" << std::endl;
    return ms;
}

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

    funcY2R(src1, rgb1);
    funcY2R(src2, rgb2);
    funcOverlay(rgb1, rgb2, blended, 128);
    funcR2Y(blended, out);

    Timer t;
    int frames = 0;
    int total_ops = 0;

    for (int r = 0; r < REPEAT_COUNT; ++r) {
        for (int alpha = 1; alpha <= 255; alpha += 3) {
            funcY2R(src1, rgb1);
            funcY2R(src2, rgb2);
            funcOverlay(rgb1, rgb2, blended, alpha);
            funcR2Y(blended, out);
            
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

void runDifferentialAnalysis(const std::string& label, 
                             FuncYUV2RGB_P3 funcBase, 
                             FuncYUV2RGB_P3 funcMem, 
                             FuncYUV2RGB_P3 funcComp,
                             FuncYUV2RGB_P3 funcShuffle) 
{
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "Running Analysis for: " << label << " (YUV2RGB_RGB888)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    YUVFrame input(WIDTH, HEIGHT);
    if (!readYUV("../data/dem1.yuv", input)) {
        std::cerr << "Error: dem1.yuv missing! Using empty data." << std::endl;
    }
    RGBFrame output(WIDTH, HEIGHT, RGBFormat::RGB888);

    funcBase(input, output);

    Timer t;
    int loops = 500;
    
    t.reset();
    for (int i = 0; i < loops; ++i) funcBase(input, output);
    double t_base = t.elapsed();

    t.reset();
    for (int i = 0; i < loops; ++i) funcMem(input, output);
    double t_mem = t.elapsed();

    t.reset();
    for (int i = 0; i < loops; ++i) funcComp(input, output);
    double t_comp = t.elapsed();

    double t_shuffle = 0.0;
    if (funcShuffle) {
        t.reset(); for(int i=0; i<loops; ++i) funcShuffle(input, output);
        t_shuffle = t.elapsed();
    }

    double avg_base = t_base / loops;
    
    std::cout << "Baseline:    " << t_base << " ms (" << avg_base << " ms/frame)" << std::endl;
    std::cout << "MemOnly:     " << t_mem  << " ms (" << (t_mem/t_base * 100) << "%)" << std::endl;
    std::cout << "ComputeOnly: " << t_comp << " ms (" << (t_comp/t_base * 100) << "%)" << std::endl;

    double sum_ratio = (t_mem + t_comp) / t_base;
    std::cout << "Analysis: ";
    if (t_mem > t_base * 0.9) {
        std::cout << "[MEMORY BOUND]" << std::endl;
    } else if (t_comp > t_base * 0.9) {
        std::cout << "[COMPUTE BOUND]" << std::endl;
    } else if (sum_ratio > 1.2) {
        std::cout << "[LATENCY HIDDEN] -> 计算与内存重叠得很好。" << std::endl;
    } else {
        std::cout << "[MIXED] -> 计算与内存均有瓶颈，或者存在依赖延迟。" << std::endl;
    }
    std::cout << "ShuffleOnly: " << t_shuffle << " ms (" << (t_shuffle/t_base * 100) << "%) -> Pure Data Layout Overhead" << std::endl;
        
    double t_math = t_comp - t_shuffle;
    if(t_math < 0) t_math = 0; 

    std::cout << "--------------------------------" << std::endl;
    std::cout << "Est. Math Cost: " << t_math << " ms (" << (t_math/t_base * 100) << "%)" << std::endl;
    std::cout << "Est. Shuffle Cost: " << t_shuffle << " ms" << std::endl;
    
    if (t_shuffle > t_math) {
        std::cout << "Conclusion: SHUFFLE BOUND (Data permutation costs more than color math!)" << std::endl;
    } else {
        std::cout << "Conclusion: MATH BOUND" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scalar|mmx|sse|avx>" << std::endl;
        std::cerr << "Example: " << argv[0] << " sse" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::cout << "=== Lab 4.1 SIMD Benchmark (" << mode << ") | Repeat: " << REPEAT_COUNT << "x ===" << std::endl;
    
    // --- 1. Run Scalar ---
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
    // --- 4. Run AVX ---
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
    // --- 5. Run Analysis Mode ---
    else if (mode == "analyze") {
        std::cout << "=== Differential Analysis Mode: YUV2RGB_RGB888 ===" << std::endl;

        // 1. Analyze Scalar
        runDifferentialAnalysis("Scalar", 
            Scalar::YUV2RGB_RGB888, 
            Scalar::YUV2RGB_RGB888_MemOnly, 
            Scalar::YUV2RGB_RGB888_ComputeOnly,
            Scalar::YUV2RGB_RGB888_ShuffleOnly
        );

        // 2. Analyze MMX
        runDifferentialAnalysis("MMX", 
            MMX::YUV2RGB_RGB888, 
            MMX::YUV2RGB_RGB888_MemOnly, 
            MMX::YUV2RGB_RGB888_ComputeOnly,
            MMX::YUV2RGB_RGB888_ShuffleOnly
        );

        // 3. Analyze SSE
        runDifferentialAnalysis("SSE2", 
            SSE::YUV2RGB_RGB888, 
            SSE::YUV2RGB_RGB888_MemOnly, 
            SSE::YUV2RGB_RGB888_ComputeOnly,
            SSE::YUV2RGB_RGB888_ShuffleOnly
        );

        // 4. Analyze AVX
        runDifferentialAnalysis("AVX2", 
            AVX::YUV2RGB_RGB888, 
            AVX::YUV2RGB_RGB888_MemOnly, 
            AVX::YUV2RGB_RGB888_ComputeOnly,
            AVX::YUV2RGB_RGB888_ShuffleOnly
        );
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}