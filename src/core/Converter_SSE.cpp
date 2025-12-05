#include "Converter.h"
#include <emmintrin.h> // SSE2 头文件
#include <algorithm>

namespace SSE
{
    // =========================================================
    // Part 2: Alpha Blending (SSE2)
    // 逻辑: Pixel = (Pixel * alpha) >> 8
    // =========================================================
    void AlphaBlend(RGBFrame &img, uint8_t alpha)
    {
        // ARGB8888: 4 bytes/pixel
        int totalBytes = img.width * img.height * 4;
        uint8_t *data = img.data;

        // 1. 准备 Alpha 向量
        // 我们需要把 8位的 alpha 扩展为 16位，以便进行乘法运算
        // _mm_set1_epi16 会把 alpha 复制 8 次填满 128 位寄存器
        __m128i alpha_vec = _mm_set1_epi16((short)alpha);
        __m128i zero = _mm_setzero_si128();

        int i = 0;
        // SSE2 寄存器宽 128位 (16字节)，所以步长为 16
        for (; i <= totalBytes - 16; i += 16)
        {
            // 2. Load (加载 16 字节)
            // 因为使用了 AlignedAllocator，理论上可以使用 _mm_load_si128 (对齐加载)
            // 但为了安全起见（防止某些特殊宽度导致行首不对齐），这里先用 _mm_loadu_si128 (非对齐)
            __m128i src = _mm_loadu_si128((__m128i *)(data + i));

            // 3. Unpack (8位 -> 16位)
            // 乘法会导致 8位溢出，必须扩展到 16位。
            // src (16字节) 被拆分为:
            // lo (低8字节扩展为16字节) 和 hi (高8字节扩展为16字节)
            __m128i src_lo = _mm_unpacklo_epi8(src, zero);
            __m128i src_hi = _mm_unpackhi_epi8(src, zero);

            // 4. Multiply (16位乘法)
            src_lo = _mm_mullo_epi16(src_lo, alpha_vec);
            src_hi = _mm_mullo_epi16(src_hi, alpha_vec);

            // 5. Shift (除以 256)
            src_lo = _mm_srli_epi16(src_lo, 8);
            src_hi = _mm_srli_epi16(src_hi, 8);

            // 6. Pack (16位 -> 8位)
            // _mm_packus_epi16: 将两个 16位 向量打包回一个 8位 向量
            // "us" 表示 Unsigned Saturation (无符号饱和)，会自动将负数变为0，超过255变为255
            __m128i result = _mm_packus_epi16(src_lo, src_hi);

            // 7. Store
            _mm_storeu_si128((__m128i *)(data + i), result);
        }

        // 8. 尾部处理 (Scalar fallback)
        for (; i < totalBytes; ++i)
        {
            data[i] = (data[i] * alpha) >> 8;
        }
    }

    // =========================================================
    // Part 3: Image Overlay (SSE2)
    // 逻辑: Dst = (Src1 * (256-a) + Src2 * a) >> 8
    // =========================================================
    void ImageOverlay(const RGBFrame &src1, const RGBFrame &src2, RGBFrame &dst, uint8_t alpha)
    {
        // RGB888: 3 bytes/pixel
        // SSE 处理不关心像素结构，只关心字节流，所以即使是 RGB 也可以按 16 字节块处理
        int totalBytes = src1.width * src1.height * 3;

        const uint8_t *pSrc1 = src1.data;
        const uint8_t *pSrc2 = src2.data;
        uint8_t *pDst = dst.data;

        // 准备权重
        short w1_val = 256 - alpha;
        short w2_val = alpha;

        __m128i w1_vec = _mm_set1_epi16(w1_val);
        __m128i w2_vec = _mm_set1_epi16(w2_val);
        __m128i zero = _mm_setzero_si128();

        int i = 0;
        for (; i <= totalBytes - 16; i += 16)
        {
            // 1. Load (分别加载 src1 和 src2)
            __m128i s1 = _mm_loadu_si128((const __m128i *)(pSrc1 + i));
            __m128i s2 = _mm_loadu_si128((const __m128i *)(pSrc2 + i));

            // 2. Unpack (扩展到 16位)
            __m128i s1_lo = _mm_unpacklo_epi8(s1, zero);
            __m128i s1_hi = _mm_unpackhi_epi8(s1, zero);

            __m128i s2_lo = _mm_unpacklo_epi8(s2, zero);
            __m128i s2_hi = _mm_unpackhi_epi8(s2, zero);

            // 3. Calc (Src1 * w1 + Src2 * w2)
            // 注意：SSE2 支持并行加法和并行乘法
            s1_lo = _mm_mullo_epi16(s1_lo, w1_vec);
            s1_hi = _mm_mullo_epi16(s1_hi, w1_vec);

            s2_lo = _mm_mullo_epi16(s2_lo, w2_vec);
            s2_hi = _mm_mullo_epi16(s2_hi, w2_vec);

            __m128i res_lo = _mm_add_epi16(s1_lo, s2_lo);
            __m128i res_hi = _mm_add_epi16(s1_hi, s2_hi);

            // 4. Shift (>> 8)
            res_lo = _mm_srli_epi16(res_lo, 8);
            res_hi = _mm_srli_epi16(res_hi, 8);

            // 5. Pack & Store
            __m128i result = _mm_packus_epi16(res_lo, res_hi);
            _mm_storeu_si128((__m128i *)(pDst + i), result);
        }

        // 尾部处理
        for (; i < totalBytes; ++i)
        {
            int val = (pSrc1[i] * w1_val + pSrc2[i] * w2_val) >> 8;
            pDst[i] = (uint8_t)val;
        }
    }

    // =========================================================
    // Part 4: YUV420 -> ARGB8888 (SSE2)
    // 优化策略:
    // 1. 一次处理 16 个像素 (Y: 16字节, U/V: 8字节)
    // 2. 使用 Shift 6 系数避免 16位乘法溢出
    // 3. 使用 Unpack 指令巧妙进行 U/V 上采样和 ARGB 数据交织
    // =========================================================
    void YUV2RGB_ARGB8888(const YUVFrame &src, RGBFrame &dst, uint8_t alpha)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // --- 1. 准备常量系数 (Shift 6) ---
        // R = Y + ((90 * V) >> 6)
        // G = Y - ((22 * U + 46 * V) >> 6)
        // B = Y + ((113 * U) >> 6)
        __m128i c90 = _mm_set1_epi16(90);
        __m128i c22 = _mm_set1_epi16(22);
        __m128i c46 = _mm_set1_epi16(46);
        __m128i c113 = _mm_set1_epi16(113);

        __m128i zero = _mm_setzero_si128();
        __m128i c128 = _mm_set1_epi16(128);
        __m128i alphaVec = _mm_set1_epi8(alpha);

        for (int y = 0; y < height; y++)
        {
            // UV 的行索引是 y / 2
            int uvOffset = (y / 2) * (width / 2);
            
            // 每次循环处理 16 个像素
            for (int x = 0; x < width; x += 16)
            {
                // --- Step 1: 加载数据 ---
                
                // 加载 16 个 Y (128位)
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));

                // 加载 8 个 U 和 8 个 V (64位)
                // _mm_loadl_epi64 加载 64位 到寄存器低半部分，高半部分清零
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // --- Step 2: 上采样 U/V (8 -> 16) ---
                // 利用 unpacklo_epi8(x, x) 将 [u0, u1, u2...] 变成 [u0, u0, u1, u1, u2, u2...]
                // 这完美实现了“最近邻”水平 2倍 上采样
                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                // --- Step 3: 扩展为 16位 并 减去 128 ---
                // Y 分为低8个(y_lo) 和 高8个(y_hi)
                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);

                // U, V 也分为低8个和高8个，并减去 128
                __m128i u_lo = _mm_sub_epi16(_mm_unpacklo_epi8(u_raw, zero), c128);
                __m128i u_hi = _mm_sub_epi16(_mm_unpackhi_epi8(u_raw, zero), c128);

                __m128i v_lo = _mm_sub_epi16(_mm_unpacklo_epi8(v_raw, zero), c128);
                __m128i v_hi = _mm_sub_epi16(_mm_unpackhi_epi8(v_raw, zero), c128);

                // --- Step 4: 计算 RGB (16位运算) ---
                // 使用算术右移 (srai) 保持符号

                // R = Y + (90 * V >> 6)
                __m128i r_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c90, v_lo), 6));
                __m128i r_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c90, v_hi), 6));

                // B = Y + (113 * U >> 6)
                __m128i b_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c113, u_lo), 6));
                __m128i b_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c113, u_hi), 6));

                // G = Y - ((22 * U + 46 * V) >> 6)
                __m128i g_part_lo = _mm_add_epi16(_mm_mullo_epi16(c22, u_lo), _mm_mullo_epi16(c46, v_lo));
                __m128i g_part_hi = _mm_add_epi16(_mm_mullo_epi16(c22, u_hi), _mm_mullo_epi16(c46, v_hi));
                
                __m128i g_lo = _mm_sub_epi16(y_lo, _mm_srai_epi16(g_part_lo, 6));
                __m128i g_hi = _mm_sub_epi16(y_hi, _mm_srai_epi16(g_part_hi, 6));

                // --- Step 5: Pack 回 8位 (包含饱和处理) ---
                // 此时 R, G, B 各自包含 16 个像素的连续数据
                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec; // 16 个 Alpha

                // --- Step 6: 交织 (Interleave) 为 ARGB ---
                // 目前: R=[R0...R15], G=[G0...G15], B=[B0...B15], A=[A0...A15]
                // 目标: [B0 G0 R0 A0 | B1 G1 R1 A1 ...]

                // 1. 合并 B 和 G -> BG
                // BG_lo = [B0 G0 B1 G1 ... B7 G7]
                // BG_hi = [B8 G8 B9 G9 ... B15 G15]
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);

                // 2. 合并 R 和 A -> RA
                // RA_lo = [R0 A0 R1 A1 ... R7 A7]
                // RA_hi = [R8 A8 R9 A9 ... R15 A15]
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                // 3. 合并 BG 和 RA -> BGRA
                // 将 16位的 BG 对 和 16位的 RA 对交织
                // result_0 = [B0 G0 R0 A0 | B1 G1 R1 A1 | ... | B3 G3 R3 A3] (前4个像素)
                __m128i result_0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i result_1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i result_2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i result_3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // --- Step 7: 存储 (64 字节) ---
                __m128i *d = (__m128i *)(dstPtr + (y * width + x) * 4);
                _mm_storeu_si128(d + 0, result_0);
                _mm_storeu_si128(d + 1, result_1);
                _mm_storeu_si128(d + 2, result_2);
                _mm_storeu_si128(d + 3, result_3);
            }
        }
    }

    // =========================================================
    // Part 5: YUV420 -> RGB888 (SSE2)
    // 难点: 目标是 3字节/像素 (BGR)，无法直接对齐 SSE 写入。
    // 策略:
    // 1. 计算逻辑与 ARGB8888 完全一致 (生成 BGRA)。
    // 2. 将结果暂存到栈上的临时 buffer。
    // 3. 使用 scalar 循环进行 "Overlap Write" (重叠写入) 剔除 Alpha。
    // =========================================================
    void YUV2RGB_RGB888(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // --- 1. 准备系数 (Shift 6) ---
        __m128i c90 = _mm_set1_epi16(90);
        __m128i c22 = _mm_set1_epi16(22);
        __m128i c46 = _mm_set1_epi16(46);
        __m128i c113 = _mm_set1_epi16(113);

        __m128i zero = _mm_setzero_si128();
        __m128i c128 = _mm_set1_epi16(128);
        // Alpha 设为 0 或 255 都可以，反正会被丢弃
        __m128i alphaVec = _mm_set1_epi8(0); 

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 16)
            {
                // ... (这部分加载和计算逻辑与 YUV2RGB_ARGB8888 完全相同) ...
                
                // 1. Load
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // 2. Upsample UV
                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                // 3. Unpack & Sub 128
                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);
                __m128i u_lo = _mm_sub_epi16(_mm_unpacklo_epi8(u_raw, zero), c128);
                __m128i u_hi = _mm_sub_epi16(_mm_unpackhi_epi8(u_raw, zero), c128);
                __m128i v_lo = _mm_sub_epi16(_mm_unpacklo_epi8(v_raw, zero), c128);
                __m128i v_hi = _mm_sub_epi16(_mm_unpackhi_epi8(v_raw, zero), c128);

                // 4. Calculate
                __m128i r_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c90, v_lo), 6));
                __m128i r_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c90, v_hi), 6));

                __m128i b_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c113, u_lo), 6));
                __m128i b_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c113, u_hi), 6));

                __m128i g_part_lo = _mm_add_epi16(_mm_mullo_epi16(c22, u_lo), _mm_mullo_epi16(c46, v_lo));
                __m128i g_part_hi = _mm_add_epi16(_mm_mullo_epi16(c22, u_hi), _mm_mullo_epi16(c46, v_hi));
                __m128i g_lo = _mm_sub_epi16(y_lo, _mm_srai_epi16(g_part_lo, 6));
                __m128i g_hi = _mm_sub_epi16(y_hi, _mm_srai_epi16(g_part_hi, 6));

                // 5. Pack
                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec;

                // 6. Interleave (Planar -> BGRA Packed)
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                __m128i res0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i res1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i res2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i res3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // --- 核心变化 Step 7: Overlap Write ---
                // 我们手头有 4 个寄存器，共 16 个像素 (每个像素 4 字节 BGRA)
                // 我们需要写入 16 个像素 (每个像素 3 字节 BGR)
                
                // 将结果暂存到栈上 (16个 int)
                uint32_t buffer[16]; 
                _mm_storeu_si128((__m128i*)&buffer[0], res0);
                _mm_storeu_si128((__m128i*)&buffer[4], res1);
                _mm_storeu_si128((__m128i*)&buffer[8], res2);
                _mm_storeu_si128((__m128i*)&buffer[12], res3);

                // 计算目标地址
                uint8_t *ptr = dstPtr + (y * width + x) * 3;

                // 循环写入
                // 这里的技巧是：每次写入一个 int (4字节)，覆盖了 B G R A
                // 下一次写入时，指针只移动 3 字节，于是新像素的 B 覆盖了上一个像素的 A
                for (int k = 0; k < 16; ++k) {
                    // 允许非对齐写入 (x86 支持)
                    *(uint32_t*)(ptr + k * 3) = buffer[k];
                }
            }
        }
    }

    // 辅助函数：水平相加并提取 (SSE2版)
    // 将 __m128i 中的 32位整数水平相加
    // 比如 [A, B, C, D] -> A+B+C+D
    static inline int32_t hadd_epi32_sum(__m128i reg) {
        // reg = [A, B, C, D]
        __m128i high = _mm_unpackhi_epi64(reg, reg); // [C, D, C, D]
        __m128i sum  = _mm_add_epi32(reg, high);     // [A+C, B+D, ...]
        
        // sum 现在低位是 A+C, 高位是 B+D (在低64位里)
        // 移位再加一次
        __m128i sum_high = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1)); // splat B+D
        sum = _mm_add_epi32(sum, sum_high); // (A+C) + (B+D)
        
        return _mm_cvtsi128_si32(sum);
    }

    // =========================================================
    // Part 6: ARGB8888 -> YUV420 (SSE2)
    // 逻辑: 
    // 1. 每次处理 2行 x 8列 (2x2 block 方便下采样)
    // 2. 利用 _mm_madd_epi16 快速计算点积
    // =========================================================
    void RGB2YUV_ARGB8888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        // Y 系数: R*77, G*150, B*29
        // 顺序注意：我们解包后通常是 B G R A
        // madd_epi16 是相邻相乘再相加: (p0*c0 + p1*c1), (p2*c2 + p3*c3) ...
        // 像素: B G R A
        // 系数: 29 150 77 0
        // Res: (B*29 + G*150) + (R*77 + A*0)
        __m128i cY = _mm_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0);

        // U 系数: R*-43, G*-84, B*127 + 128
        // Coeff: 127, -84, -43, 0
        __m128i cU = _mm_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0);

        // V 系数: R*127, G*-107, B*-20 + 128
        // Coeff: -20, -107, 127, 0
        __m128i cV = _mm_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0);

        __m128i zero = _mm_setzero_si128();

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 8)
            {
                // 我们需要处理 Row y 和 Row y+1，每行 8 个像素
                // 结果：16 个 Y，4 个 U，4 个 V

                // --- 1. 计算 Y (逐行处理) ---
                for (int r = 0; r < 2; ++r) 
                {
                    int currY = y + r;
                    const uint8_t* pSrc = src.data + (currY * width + x) * 4;
                    uint8_t* pDstY = dst.Y + currY * width + x;

                    // 加载 8 个像素 (32 字节 = 2个寄存器)
                    // px1 = [P0, P1, P2, P3], px2 = [P4, P5, P6, P7]
                    __m128i px1 = _mm_loadu_si128((const __m128i*)pSrc);
                    __m128i px2 = _mm_loadu_si128((const __m128i*)(pSrc + 16));

                    // 解包成 16位: B G R A
                    // p0 = P0, P1; p1 = P2, P3 ...
                    __m128i p0 = _mm_unpacklo_epi8(px1, zero);
                    __m128i p1 = _mm_unpackhi_epi8(px1, zero);
                    __m128i p2 = _mm_unpacklo_epi8(px2, zero);
                    __m128i p3 = _mm_unpackhi_epi8(px2, zero);

                    // 计算 Y
                    // madd 之后得到 [SumBG, SumRA, SumBG, SumRA ...] (32位)
                    // 我们需要把相邻的 SumBG + SumRA 加起来
                    
                    // P0, P1
                    __m128i y0_parts = _mm_madd_epi16(p0, cY); 
                    // P2, P3
                    __m128i y1_parts = _mm_madd_epi16(p1, cY);
                    // P4, P5
                    __m128i y2_parts = _mm_madd_epi16(p2, cY);
                    // P6, P7
                    __m128i y3_parts = _mm_madd_epi16(p3, cY);

                    // 水平相加得到最终 Y 值 (32位)
                    // _mm_hadd_epi32 (需要 SSE3, 这里用 SSE2 模拟或直接 shuffle)
                    // 实际上 madd 后的结果是 [Y0_part1, Y0_part2, Y1_part1, Y1_part2]
                    // 我们想要 [Y0, Y1, Y2, Y3]
                    
                    // 技巧: Shuffle + Add
                    // p0_sum = [Y0, Y1] (包含在两个 32位里)
                    // 这是一个通用写法:
                    // val = [A, B, C, D], swapped = [B, A, D, C] -> add -> [A+B, ...]
                    
                    __m128i y01 = _mm_add_epi32(y0_parts, _mm_shuffle_epi32(y0_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y23 = _mm_add_epi32(y1_parts, _mm_shuffle_epi32(y1_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y45 = _mm_add_epi32(y2_parts, _mm_shuffle_epi32(y2_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y67 = _mm_add_epi32(y3_parts, _mm_shuffle_epi32(y3_parts, _MM_SHUFFLE(2, 3, 0, 1)));

                    // 现在 y01 的第0和第2个 32位 包含了 Y0, Y1 的值 (虽然还没右移)
                    // 我们需要把它们收集起来 Pack 成 16位 -> 8位
                    // 这一步比较繁琐，为了不写太长，我们先把结果存到 buffer
                    // 或者使用 scalar 处理最后一步 (Shift + Clamp)
                    
                    // 简单粗暴法：直接提取
                    // 注意：Shuffle 后 y01 = [Y0, Y0, Y1, Y1]
                    int val0 = _mm_cvtsi128_si32(y01) >> 8;
                    int val1 = _mm_cvtsi128_si32(_mm_srli_si128(y01, 8)) >> 8;
                    int val2 = _mm_cvtsi128_si32(y23) >> 8;
                    int val3 = _mm_cvtsi128_si32(_mm_srli_si128(y23, 8)) >> 8;
                    int val4 = _mm_cvtsi128_si32(y45) >> 8;
                    int val5 = _mm_cvtsi128_si32(_mm_srli_si128(y45, 8)) >> 8;
                    int val6 = _mm_cvtsi128_si32(y67) >> 8;
                    int val7 = _mm_cvtsi128_si32(_mm_srli_si128(y67, 8)) >> 8;
                    
                    pDstY[0] = std::clamp(val0, 0, 255);
                    pDstY[1] = std::clamp(val1, 0, 255);
                    pDstY[2] = std::clamp(val2, 0, 255);
                    pDstY[3] = std::clamp(val3, 0, 255);
                    pDstY[4] = std::clamp(val4, 0, 255);
                    pDstY[5] = std::clamp(val5, 0, 255);
                    pDstY[6] = std::clamp(val6, 0, 255);
                    pDstY[7] = std::clamp(val7, 0, 255);
                }

                // --- 2. 计算 UV (2x2 均值) ---
                // Row 0 data
                __m128i r0_p1 = _mm_loadu_si128((const __m128i*)(src.data + (y * width + x) * 4));
                __m128i r0_p2 = _mm_loadu_si128((const __m128i*)(src.data + (y * width + x) * 4 + 16));
                
                // Row 1 data
                __m128i r1_p1 = _mm_loadu_si128((const __m128i*)(src.data + ((y + 1) * width + x) * 4));
                __m128i r1_p2 = _mm_loadu_si128((const __m128i*)(src.data + ((y + 1) * width + x) * 4 + 16));

                // 我们要计算 4 个 block 的均值
                // Block 0: r0_p1 的前两个 + r1_p1 的前两个
                // 这意味着我们需要先把它们转成 16位 然后相加
                
                __m128i r0_lo = _mm_unpacklo_epi8(r0_p1, zero); // P0, P1
                __m128i r0_hi = _mm_unpackhi_epi8(r0_p1, zero); // P2, P3
                __m128i r0_lo2 = _mm_unpacklo_epi8(r0_p2, zero); // P4, P5
                __m128i r0_hi2 = _mm_unpackhi_epi8(r0_p2, zero); // P6, P7

                __m128i r1_lo = _mm_unpacklo_epi8(r1_p1, zero);
                __m128i r1_hi = _mm_unpackhi_epi8(r1_p1, zero);
                __m128i r1_lo2 = _mm_unpacklo_epi8(r1_p2, zero);
                __m128i r1_hi2 = _mm_unpackhi_epi8(r1_p2, zero);

                // Sum vertical (Row0 + Row1)
                __m128i sum01 = _mm_add_epi16(r0_lo, r1_lo); // P0+P0', P1+P1'
                __m128i sum23 = _mm_add_epi16(r0_hi, r1_hi);
                __m128i sum45 = _mm_add_epi16(r0_lo2, r1_lo2);
                __m128i sum67 = _mm_add_epi16(r0_hi2, r1_hi2);

                // Sum horizontal (P0+P1, P2+P3...)
                // 这里的 P0+P1 指的是 [B0+B1, G0+G1, R0+R1, A0+A1]
                // 我们可以使用 shuffle 或者 hadd (SSE3). 在 SSE2 中：
                // sum01 = [Px0_sum, Px1_sum]
                // 我们想让 Px0_sum 和 Px1_sum 相加
                // shift 64位 (8字节, 即一个扩充后的像素大小)
                __m128i avg0 = _mm_add_epi16(sum01, _mm_bsrli_si128(sum01, 8)); // Low 64bit has Sum(Block0)
                __m128i avg1 = _mm_add_epi16(sum23, _mm_bsrli_si128(sum23, 8));
                __m128i avg2 = _mm_add_epi16(sum45, _mm_bsrli_si128(sum45, 8));
                __m128i avg3 = _mm_add_epi16(sum67, _mm_bsrli_si128(sum67, 8));

                // 此时 avgX 的低 64位 包含了 Block X 的 (B_sum, G_sum, R_sum, A_sum)
                // Divide by 4
                avg0 = _mm_srli_epi16(avg0, 2);
                avg1 = _mm_srli_epi16(avg1, 2);
                avg2 = _mm_srli_epi16(avg2, 2);
                avg3 = _mm_srli_epi16(avg3, 2);

                // Calc U, V
                // 利用 madd 
                // U = (madd(avg, cU) >> 8) + 128
                int u0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cU)) >> 8) + 128;
                int u1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cU)) >> 8) + 128;
                int u2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cU)) >> 8) + 128;
                int u3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cU)) >> 8) + 128;

                int v0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cV)) >> 8) + 128;
                int v1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cV)) >> 8) + 128;
                int v2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cV)) >> 8) + 128;
                int v3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cV)) >> 8) + 128;

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                dst.U[uvIdx + 0] = std::clamp(u0, 0, 255);
                dst.U[uvIdx + 1] = std::clamp(u1, 0, 255);
                dst.U[uvIdx + 2] = std::clamp(u2, 0, 255);
                dst.U[uvIdx + 3] = std::clamp(u3, 0, 255);

                dst.V[uvIdx + 0] = std::clamp(v0, 0, 255);
                dst.V[uvIdx + 1] = std::clamp(v1, 0, 255);
                dst.V[uvIdx + 2] = std::clamp(v2, 0, 255);
                dst.V[uvIdx + 3] = std::clamp(v3, 0, 255);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 辅助函数: 加载 4 个 RGB 像素 (12字节) -> 转换为 4 个 32位整数
    // 纯 SSE2 优化版: 1次 Load + 移位拼接，消除标量加载开销
    // -----------------------------------------------------------------------
    static inline __m128i load_rgb_4px(const uint8_t* ptr) {
        // 1. 加载 16 字节 (包含 4 个像素 + 4 字节冗余，允许越界读取只要不跨页)
        // 这里的假设是图像 buffer 分配时如 ImageTypes.h 中所示有 padding，否则最后几个像素可能越界
        __m128i raw = _mm_loadu_si128((const __m128i*)ptr);

        // 我们需要提取 4 个像素，起始偏移分别为 0, 3, 6, 9 字节
        // SSE2 只有字节级别的立即数移位 _mm_srli_si128
        
        // P0 (Byte 0..2): 原位，不需要移
        __m128i p0 = raw; 

        // P1 (Byte 3..5): 整体右移 3 字节
        __m128i p1 = _mm_srli_si128(raw, 3);

        // P2 (Byte 6..8): 整体右移 6 字节
        __m128i p2 = _mm_srli_si128(raw, 6);

        // P3 (Byte 9..11): 整体右移 9 字节
        __m128i p3 = _mm_srli_si128(raw, 9);

        // 现在 P0, P1, P2, P3 的低 32 位包含了我们想要的数据 (虽然高位有脏数据)
        // 我们利用 Unpack 指令将它们合并
        
        // p01 = [P0_low32, P1_low32, ...]
        __m128i p01 = _mm_unpacklo_epi32(p0, p1);
        
        // p23 = [P2_low32, P3_low32, ...]
        __m128i p23 = _mm_unpacklo_epi32(p2, p3);
        
        // final = [P0, P1, P2, P3]
        __m128i final_reg = _mm_unpacklo_epi64(p01, p23);

        // 最后统一把 Alpha 通道 (高8位) 清零
        // RGB (3字节) 读入变成 int 后，最高位是脏数据
        __m128i mask = _mm_set1_epi32(0x00FFFFFF);
        return _mm_and_si128(final_reg, mask);
    }

    // =========================================================
    // Part 7: RGB888 -> YUV420 (SSE2)
    // 逻辑: 
    // 1. 使用 load_rgb_4px 将 RGB 转换为 "ARGB" (A=0) 格式。
    // 2. 后续计算逻辑与 ARGB8888 -> YUV 完全一致。
    // =========================================================
    void RGB2YUV_RGB888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        // 系数定义 (完全复用 ARGB 版)
        // Y: (29*B + 150*G + 77*R)
        __m128i cY = _mm_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0);
        // U: (127*B - 84*G - 43*R) + 128
        __m128i cU = _mm_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0);
        // V: (-20*B - 107*G + 127*R) + 128
        __m128i cV = _mm_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0);

        __m128i zero = _mm_setzero_si128();

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 8)
            {
                // --- 1. 计算 Y (逐行处理) ---
                for (int r = 0; r < 2; ++r) 
                {
                    int currY = y + r;
                    const uint8_t* pSrc = src.data + (currY * width + x) * 3; // 注意步长是 3
                    uint8_t* pDstY = dst.Y + currY * width + x;

                    // 【核心变化点】：使用 helper 加载数据
                    // px1 = Pixels [0, 1, 2, 3]
                    __m128i px1 = load_rgb_4px(pSrc);
                    // px2 = Pixels [4, 5, 6, 7] (偏移 4个像素 = 12字节)
                    __m128i px2 = load_rgb_4px(pSrc + 12);

                    // Unpack (8位 -> 16位)
                    __m128i p0 = _mm_unpacklo_epi8(px1, zero);
                    __m128i p1 = _mm_unpackhi_epi8(px1, zero);
                    __m128i p2 = _mm_unpacklo_epi8(px2, zero);
                    __m128i p3 = _mm_unpackhi_epi8(px2, zero);

                    // Madd 计算 (点积)
                    __m128i y0_parts = _mm_madd_epi16(p0, cY); 
                    __m128i y1_parts = _mm_madd_epi16(p1, cY);
                    __m128i y2_parts = _mm_madd_epi16(p2, cY);
                    __m128i y3_parts = _mm_madd_epi16(p3, cY);

                    // 收集结果 (Shuffle + Add)
                    __m128i y01 = _mm_add_epi32(y0_parts, _mm_shuffle_epi32(y0_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y23 = _mm_add_epi32(y1_parts, _mm_shuffle_epi32(y1_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y45 = _mm_add_epi32(y2_parts, _mm_shuffle_epi32(y2_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y67 = _mm_add_epi32(y3_parts, _mm_shuffle_epi32(y3_parts, _MM_SHUFFLE(2, 3, 0, 1)));

                    // 提取并存储 (Scalar Store)
                    int val0 = _mm_cvtsi128_si32(y01) >> 8;
                    int val1 = _mm_cvtsi128_si32(_mm_srli_si128(y01, 8)) >> 8;
                    int val2 = _mm_cvtsi128_si32(y23) >> 8;
                    int val3 = _mm_cvtsi128_si32(_mm_srli_si128(y23, 8)) >> 8;
                    int val4 = _mm_cvtsi128_si32(y45) >> 8;
                    int val5 = _mm_cvtsi128_si32(_mm_srli_si128(y45, 8)) >> 8;
                    int val6 = _mm_cvtsi128_si32(y67) >> 8;
                    int val7 = _mm_cvtsi128_si32(_mm_srli_si128(y67, 8)) >> 8;
                    
                    pDstY[0] = std::clamp(val0, 0, 255);
                    pDstY[1] = std::clamp(val1, 0, 255);
                    pDstY[2] = std::clamp(val2, 0, 255);
                    pDstY[3] = std::clamp(val3, 0, 255);
                    pDstY[4] = std::clamp(val4, 0, 255);
                    pDstY[5] = std::clamp(val5, 0, 255);
                    pDstY[6] = std::clamp(val6, 0, 255);
                    pDstY[7] = std::clamp(val7, 0, 255);
                }

                // --- 2. 计算 UV (2x2 均值) ---
                // Row 0 data
                __m128i r0_p1 = load_rgb_4px(src.data + (y * width + x) * 3);
                __m128i r0_p2 = load_rgb_4px(src.data + (y * width + x) * 3 + 12);
                
                // Row 1 data
                __m128i r1_p1 = load_rgb_4px(src.data + ((y + 1) * width + x) * 3);
                __m128i r1_p2 = load_rgb_4px(src.data + ((y + 1) * width + x) * 3 + 12);

                // 后续逻辑与 ARGB 版完全一致 (Unpack -> Sum -> Avg -> Calc)
                
                __m128i r0_lo = _mm_unpacklo_epi8(r0_p1, zero);
                __m128i r0_hi = _mm_unpackhi_epi8(r0_p1, zero);
                __m128i r0_lo2 = _mm_unpacklo_epi8(r0_p2, zero);
                __m128i r0_hi2 = _mm_unpackhi_epi8(r0_p2, zero);

                __m128i r1_lo = _mm_unpacklo_epi8(r1_p1, zero);
                __m128i r1_hi = _mm_unpackhi_epi8(r1_p1, zero);
                __m128i r1_lo2 = _mm_unpacklo_epi8(r1_p2, zero);
                __m128i r1_hi2 = _mm_unpackhi_epi8(r1_p2, zero);

                // Sum vertical
                __m128i sum01 = _mm_add_epi16(r0_lo, r1_lo);
                __m128i sum23 = _mm_add_epi16(r0_hi, r1_hi);
                __m128i sum45 = _mm_add_epi16(r0_lo2, r1_lo2);
                __m128i sum67 = _mm_add_epi16(r0_hi2, r1_hi2);

                // Sum horizontal (Shift 8 bytes = 2 pixels = 64 bit)
                __m128i avg0 = _mm_add_epi16(sum01, _mm_bsrli_si128(sum01, 8));
                __m128i avg1 = _mm_add_epi16(sum23, _mm_bsrli_si128(sum23, 8));
                __m128i avg2 = _mm_add_epi16(sum45, _mm_bsrli_si128(sum45, 8));
                __m128i avg3 = _mm_add_epi16(sum67, _mm_bsrli_si128(sum67, 8));

                // Divide by 4
                avg0 = _mm_srli_epi16(avg0, 2);
                avg1 = _mm_srli_epi16(avg1, 2);
                avg2 = _mm_srli_epi16(avg2, 2);
                avg3 = _mm_srli_epi16(avg3, 2);

                // Calc U, V
                int u0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cU)) >> 8) + 128;
                int u1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cU)) >> 8) + 128;
                int u2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cU)) >> 8) + 128;
                int u3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cU)) >> 8) + 128;

                int v0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cV)) >> 8) + 128;
                int v1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cV)) >> 8) + 128;
                int v2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cV)) >> 8) + 128;
                int v3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cV)) >> 8) + 128;

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                dst.U[uvIdx + 0] = std::clamp(u0, 0, 255);
                dst.U[uvIdx + 1] = std::clamp(u1, 0, 255);
                dst.U[uvIdx + 2] = std::clamp(u2, 0, 255);
                dst.U[uvIdx + 3] = std::clamp(u3, 0, 255);

                dst.V[uvIdx + 0] = std::clamp(v0, 0, 255);
                dst.V[uvIdx + 1] = std::clamp(v1, 0, 255);
                dst.V[uvIdx + 2] = std::clamp(v2, 0, 255);
                dst.V[uvIdx + 3] = std::clamp(v3, 0, 255);
            }
        }
    }
}