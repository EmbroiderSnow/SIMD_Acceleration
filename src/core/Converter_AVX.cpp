#include "Converter.h"
#include <immintrin.h>
#include <algorithm>

namespace AVX
{
    // =========================================================
    // Alpha Blend (AVX2)
    // Width: 256 bit (32 bytes)
    // =========================================================
    void AlphaBlend(RGBFrame &img, uint8_t alpha)
    {
        int totalBytes = img.width * img.height * 4;
        uint8_t *data = img.data;

        // Prepare Constants (Extended to 256-bit)
        __m256i alpha_vec = _mm256_set1_epi16((short)alpha);
        __m256i zero = _mm256_setzero_si256();

        int i = 0;
        // Stride 32 (AVX2 processes 32 bytes)
        for (; i <= totalBytes - 32; i += 32)
        {
            // Load (32 bytes)
            __m256i src = _mm256_loadu_si256((const __m256i *)(data + i));

            // Unpack (8-bit -> 16-bit)
            // Note: AVX2 unpack operates within 128-bit lanes
            __m256i src_lo = _mm256_unpacklo_epi8(src, zero);
            __m256i src_hi = _mm256_unpackhi_epi8(src, zero);

            // Multiply (16-bit)
            src_lo = _mm256_mullo_epi16(src_lo, alpha_vec);
            src_hi = _mm256_mullo_epi16(src_hi, alpha_vec);

            // Shift (>> 8)
            src_lo = _mm256_srli_epi16(src_lo, 8);
            src_hi = _mm256_srli_epi16(src_hi, 8);

            // Pack (16-bit -> 8-bit)
            // Rearranges lanes automatically to restore order
            __m256i result = _mm256_packus_epi16(src_lo, src_hi);

            // Store
            _mm256_storeu_si256((__m256i *)(data + i), result);
        }

        // Scalar fallback
        for (; i < totalBytes; ++i)
        {
            data[i] = (data[i] * alpha) >> 8;
        }
    }

    // =========================================================
    // Image Overlay (AVX2)
    // =========================================================
    void ImageOverlay(const RGBFrame &src1, const RGBFrame &src2, RGBFrame &dst, uint8_t alpha)
    {
        int totalBytes = src1.width * src1.height * 3;

        const uint8_t *pSrc1 = src1.data;
        const uint8_t *pSrc2 = src2.data;
        uint8_t *pDst = dst.data;

        short w1_val = 256 - alpha;
        short w2_val = alpha;

        __m256i w1_vec = _mm256_set1_epi16(w1_val);
        __m256i w2_vec = _mm256_set1_epi16(w2_val);
        __m256i zero = _mm256_setzero_si256();

        int i = 0;
        for (; i <= totalBytes - 32; i += 32)
        {
            // 1. Load 32 bytes
            __m256i s1 = _mm256_loadu_si256((const __m256i *)(pSrc1 + i));
            __m256i s2 = _mm256_loadu_si256((const __m256i *)(pSrc2 + i));

            // 2. Unpack
            __m256i s1_lo = _mm256_unpacklo_epi8(s1, zero);
            __m256i s1_hi = _mm256_unpackhi_epi8(s1, zero);

            __m256i s2_lo = _mm256_unpacklo_epi8(s2, zero);
            __m256i s2_hi = _mm256_unpackhi_epi8(s2, zero);

            // 3. Calc (Src1 * w1 + Src2 * w2)
            s1_lo = _mm256_mullo_epi16(s1_lo, w1_vec);
            s1_hi = _mm256_mullo_epi16(s1_hi, w1_vec);

            s2_lo = _mm256_mullo_epi16(s2_lo, w2_vec);
            s2_hi = _mm256_mullo_epi16(s2_hi, w2_vec);

            __m256i res_lo = _mm256_add_epi16(s1_lo, s2_lo);
            __m256i res_hi = _mm256_add_epi16(s1_hi, s2_hi);

            // 4. Shift & Pack
            res_lo = _mm256_srli_epi16(res_lo, 8);
            res_hi = _mm256_srli_epi16(res_hi, 8);

            __m256i result = _mm256_packus_epi16(res_lo, res_hi);

            // 5. Store
            _mm256_storeu_si256((__m256i *)(pDst + i), result);
        }

        for (; i < totalBytes; ++i)
        {
            int val = (pSrc1[i] * w1_val + pSrc2[i] * w2_val) >> 8;
            pDst[i] = (uint8_t)val;
        }
    }

    // =========================================================
    // Part 4: YUV420 -> ARGB8888 (AVX2)
    // Throughput: 32 pixels per iteration (256 bit Y)
    // =========================================================
    void YUV2RGB_ARGB8888(const YUVFrame &src, RGBFrame &dst, uint8_t alpha)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // Constants (Extended to 256-bit)
        __m256i c90 = _mm256_set1_epi16(90);
        __m256i c22 = _mm256_set1_epi16(22);
        __m256i c46 = _mm256_set1_epi16(46);
        __m256i c113 = _mm256_set1_epi16(113);
        __m256i c128 = _mm256_set1_epi16(128);
        __m256i zero = _mm256_setzero_si256();
        __m256i alphaVec = _mm256_set1_epi8(alpha);

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 32)
            {
                // Load Y (32 bytes)
                __m256i y_raw = _mm256_loadu_si256((const __m256i *)(YPtr + y * width + x));

                // Load & Upsample UV (16 bytes -> 32 bytes)
                __m128i u_small = _mm_loadu_si128((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_small = _mm_loadu_si128((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // Expand 8-bit to 16-bit (Duplicate within 128-bit lane)
                __m128i u_lo_128 = _mm_unpacklo_epi8(u_small, u_small); 
                __m128i u_hi_128 = _mm_unpackhi_epi8(u_small, u_small); 
                
                // Construct 256-bit UV vectors
                __m256i u_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(u_lo_128), u_hi_128, 1);
                
                __m128i v_lo_128 = _mm_unpacklo_epi8(v_small, v_small);
                __m128i v_hi_128 = _mm_unpackhi_epi8(v_small, v_small);
                __m256i v_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(v_lo_128), v_hi_128, 1);

                // --- Process Batch 1: Pixels 0-15 (Lower 128-bit) ---
                __m256i y0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_raw));
                __m256i u0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_raw)), c128);
                __m256i v0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_raw)), c128);

                // --- Process Batch 2: Pixels 16-31 (Upper 128-bit) ---
                __m256i y1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_raw, 1));
                __m256i u1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(u_raw, 1)), c128);
                __m256i v1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(v_raw, 1)), c128);

                // RGB Calc (Batch 1)
                __m256i r0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v0), 6));
                __m256i g_part0 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u0), _mm256_mullo_epi16(c46, v0));
                __m256i g0 = _mm256_sub_epi16(y0, _mm256_srai_epi16(g_part0, 6));
                __m256i b0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u0), 6));

                // RGB Calc (Batch 2)
                __m256i r1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v1), 6));
                __m256i g_part1 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u1), _mm256_mullo_epi16(c46, v1));
                __m256i g1 = _mm256_sub_epi16(y1, _mm256_srai_epi16(g_part1, 6));
                __m256i b1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u1), 6));

                // Pack 16-bit to 8-bit
                // Note: Results are lane-separated [0-7, 16-23, 8-15, 24-31]
                __m256i R = _mm256_packus_epi16(r0, r1);
                __m256i G = _mm256_packus_epi16(g0, g1);
                __m256i B = _mm256_packus_epi16(b0, b1);
                __m256i A = alphaVec;

                // Fix Pack Order (Permute across lanes)
                // Restore linear order [0-31]
                R = _mm256_permute4x64_epi64(R, 0xD8); // Shuffle: 3, 1, 2, 0
                G = _mm256_permute4x64_epi64(G, 0xD8);
                B = _mm256_permute4x64_epi64(B, 0xD8);

                // Interleave to BGRA
                __m256i bg_lo = _mm256_unpacklo_epi8(B, G);
                __m256i bg_hi = _mm256_unpackhi_epi8(B, G);
                __m256i ra_lo = _mm256_unpacklo_epi8(R, A);
                __m256i ra_hi = _mm256_unpackhi_epi8(R, A);

                __m256i res0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                __m256i res1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);
                __m256i res2 = _mm256_unpacklo_epi16(bg_hi, ra_hi);
                __m256i res3 = _mm256_unpackhi_epi16(bg_hi, ra_hi);

                // Fix Lane Crossing & Reassemble 128-bit blocks
                // 0x20: Src1[0] | Src2[0] -> Pixels 0-7, 8-15
                __m256i out0 = _mm256_permute2x128_si256(res0, res1, 0x20); 
                __m256i out1 = _mm256_permute2x128_si256(res2, res3, 0x20); 
                // 0x31: Src1[1] | Src2[1] -> Pixels 16-23, 24-31
                __m256i out2 = _mm256_permute2x128_si256(res0, res1, 0x31); 
                __m256i out3 = _mm256_permute2x128_si256(res2, res3, 0x31); 

                // Store (128 bytes)
                __m256i *d = (__m256i *)(dstPtr + (y * width + x) * 4);
                _mm256_storeu_si256(d + 0, out0);
                _mm256_storeu_si256(d + 1, out1);
                _mm256_storeu_si256(d + 2, out2);
                _mm256_storeu_si256(d + 3, out3);
            }
        }
    }

    // =========================================================
    // Part 5: YUV420 -> RGB888 (AVX2 Shuffle Optimized)
    // Optimization: 
    // 1. Process 32 pixels per iteration (AVX2).
    // 2. Use vpshufb (Shuffle) to compress BGRA to BGR.
    // 3. Use block store strategy to resolve 3-byte write bottlenecks.
    // =========================================================
    void YUV2RGB_RGB888(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // --- 1. Prepare Constants (256-bit) ---
        __m256i c90  = _mm256_set1_epi16(90);
        __m256i c22  = _mm256_set1_epi16(22);
        __m256i c46  = _mm256_set1_epi16(46);
        __m256i c113 = _mm256_set1_epi16(113);
        __m256i c128 = _mm256_set1_epi16(128);
        __m256i zero = _mm256_setzero_si256();
        // Alpha is discarded in RGB888, set to 0
        __m256i alphaVec = zero; 

        // --- 2. Prepare Shuffle Mask (Critical) ---
        // Purpose: Discard Alpha from BGRA (4 bytes), keep BGR packed
        // Input (Per Lane): [B0 G0 R0 A0] [B1 G1 R1 A1] [B2 G2 R2 A2] [B3 G3 R3 A3]
        // Output (Per Lane): [B0 G0 R0 B1   G1 R1 B2 G2   R2 B3 G3 R3   XX XX XX XX]
        // Only the first 12 bytes per lane are valid data
        __m256i shuffleMask = _mm256_setr_epi8(
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, // Lane 0 (Low 128)
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1  // Lane 1 (High 128)
        );

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 32)
            {
                // ================== Step 1: Load & Upsample ==================
                
                // Load Y (32 pixels, 256 bits)
                __m256i y_raw = _mm256_loadu_si256((const __m256i *)(YPtr + y * width + x));

                // Load U, V (16 pixels, 128 bits)
                __m128i u_small = _mm_loadu_si128((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_small = _mm_loadu_si128((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // Upsample U, V (Double each byte: 128bit -> 256bit)
                __m128i u_lo_128 = _mm_unpacklo_epi8(u_small, u_small); 
                __m128i u_hi_128 = _mm_unpackhi_epi8(u_small, u_small);
                // Combine into 256-bit register
                __m256i u_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(u_lo_128), u_hi_128, 1);

                __m128i v_lo_128 = _mm_unpacklo_epi8(v_small, v_small);
                __m128i v_hi_128 = _mm_unpackhi_epi8(v_small, v_small);
                __m256i v_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(v_lo_128), v_hi_128, 1);

                // ================== Step 2: Convert to 16-bit & Sub 128 ==================
                
                // Batch 0: Pixels 0-15 (Use Lane 0 data)
                __m256i y0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_raw));
                __m256i u0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_raw)), c128);
                __m256i v0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_raw)), c128);

                // Batch 1: Pixels 16-31 (Use Lane 1 data)
                __m256i y1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_raw, 1));
                __m256i u1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(u_raw, 1)), c128);
                __m256i v1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(v_raw, 1)), c128);

                // ================== Step 3: Math (Integer Approximation) ==================
                
                // Calc Batch 0
                __m256i r0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v0), 6));
                __m256i g_part0 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u0), _mm256_mullo_epi16(c46, v0));
                __m256i g0 = _mm256_sub_epi16(y0, _mm256_srai_epi16(g_part0, 6));
                __m256i b0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u0), 6));

                // Calc Batch 1
                __m256i r1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v1), 6));
                __m256i g_part1 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u1), _mm256_mullo_epi16(c46, v1));
                __m256i g1 = _mm256_sub_epi16(y1, _mm256_srai_epi16(g_part1, 6));
                __m256i b1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u1), 6));

                // ================== Step 4: Pack & Permute & Interleave ==================
                
                // Pack to 8-bit (Result contains BGRA but in scrambled order due to Lanes)
                __m256i R = _mm256_packus_epi16(r0, r1);
                __m256i G = _mm256_packus_epi16(g0, g1);
                __m256i B = _mm256_packus_epi16(b0, b1);
                __m256i A = alphaVec;

                // Fix Order: 0, 2, 1, 3 (0xD8) to make it linear: 0-15, 16-31
                R = _mm256_permute4x64_epi64(R, 0xD8);
                G = _mm256_permute4x64_epi64(G, 0xD8);
                B = _mm256_permute4x64_epi64(B, 0xD8);

                // Standard Interleaving to get BGRA BGRA...
                __m256i bg_lo = _mm256_unpacklo_epi8(B, G);
                __m256i bg_hi = _mm256_unpackhi_epi8(B, G);
                __m256i ra_lo = _mm256_unpacklo_epi8(R, A);
                __m256i ra_hi = _mm256_unpackhi_epi8(R, A);

                __m256i res0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                __m256i res1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);
                __m256i res2 = _mm256_unpacklo_epi16(bg_hi, ra_hi);
                __m256i res3 = _mm256_unpackhi_epi16(bg_hi, ra_hi);

                // Fix Lane Crossing (Combine low/high lanes correctly)
                // out0 = Pixels 0-7, out1 = Pixels 8-15
                // out2 = Pixels 16-23, out3 = Pixels 24-31
                __m256i out0 = _mm256_permute2x128_si256(res0, res1, 0x20);
                __m256i out1 = _mm256_permute2x128_si256(res2, res3, 0x20);
                __m256i out2 = _mm256_permute2x128_si256(res0, res1, 0x31);
                __m256i out3 = _mm256_permute2x128_si256(res2, res3, 0x31);

                // ================== Step 5: Shuffle and Compact Store ==================
                
                // Shuffle: BGRA -> BGR_ (Inside register)
                __m256i s0 = _mm256_shuffle_epi8(out0, shuffleMask);
                __m256i s1 = _mm256_shuffle_epi8(out1, shuffleMask);
                __m256i s2 = _mm256_shuffle_epi8(out2, shuffleMask);
                __m256i s3 = _mm256_shuffle_epi8(out3, shuffleMask);

                uint8_t* p = dstPtr + (y * width + x) * 3;

                // Helper lambda: Store valid 12 bytes from a 128-bit lane
                // Input: 128-bit register (logic uses lower 128 bits)
                // Operation: Store 8 bytes (int64) + Shift 8 + Store 4 bytes (int32)
                auto store_lane_12bytes = [&](__m128i lane_data) {
                    // Store 8 bytes
                    _mm_storel_epi64((__m128i*)p, lane_data);
                    // Store remaining 4 bytes
                    *(int*)(p + 8) = _mm_cvtsi128_si32(_mm_srli_si128(lane_data, 8));
                    p += 12; // Advance pointer
                };

                // Store s0 (Pixels 0-7)
                store_lane_12bytes(_mm256_castsi256_si128(s0));         // Lane 0
                store_lane_12bytes(_mm256_extracti128_si256(s0, 1));     // Lane 1
                
                // Store s1 (Pixels 8-15)
                store_lane_12bytes(_mm256_castsi256_si128(s1));
                store_lane_12bytes(_mm256_extracti128_si256(s1, 1));

                // Store s2 (Pixels 16-23)
                store_lane_12bytes(_mm256_castsi256_si128(s2));
                store_lane_12bytes(_mm256_extracti128_si256(s2, 1));

                // Store s3 (Pixels 24-31)
                store_lane_12bytes(_mm256_castsi256_si128(s3));
                store_lane_12bytes(_mm256_extracti128_si256(s3, 1));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper Function: Compute Y/U/V Component (AVX2 Fixed)
    // Fix: Changed packus (unsigned saturation) to packs (signed saturation) 
    // to prevent negative UV values from being truncated to 0.
    // -----------------------------------------------------------------------
    static inline __m256i compute_component_avx2(__m256i p0, __m256i p1, __m256i coeffs) {
        // 1. Dot Product (Madd) -> 32-bit signed result
        __m256i res0 = _mm256_madd_epi16(p0, coeffs);
        __m256i res1 = _mm256_madd_epi16(p1, coeffs);

        // 2. Horizontal Add (Shuffle + Add) -> 32-bit result
        // Sums adjacent parts (e.g. B*c1+G*c2 and R*c3+A*0)
        __m256i sum0 = _mm256_add_epi32(res0, _mm256_shuffle_epi32(res0, _MM_SHUFFLE(2, 3, 0, 1)));
        __m256i sum1 = _mm256_add_epi32(res1, _mm256_shuffle_epi32(res1, _MM_SHUFFLE(2, 3, 0, 1)));

        // 3. Right Shift (>> 8)
        sum0 = _mm256_srai_epi32(sum0, 8);
        sum1 = _mm256_srai_epi32(sum1, 8);

        // 4. Pack 32-bit -> 16-bit
        // Using _mm256_packs_epi32 (signed saturation) is critical for U/V
        return _mm256_packs_epi32(sum0, sum1);
    }

    // =========================================================
    // ARGB8888 -> YUV420 (AVX2 Fixed)
    // =========================================================
    void RGB2YUV_ARGB8888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        __m256i cY = _mm256_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0, 29, 150, 77, 0, 29, 150, 77, 0);
        __m256i cU = _mm256_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0);
        __m256i cV = _mm256_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0);

        __m256i zero = _mm256_setzero_si256();
        __m256i offset128 = _mm256_set1_epi16(128);

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 32)
            {
                const uint8_t* r0_ptr = src.data + (y * width + x) * 4;
                const uint8_t* r1_ptr = src.data + ((y + 1) * width + x) * 4;

                __m256i r0_v0 = _mm256_loadu_si256((const __m256i*)(r0_ptr));
                __m256i r0_v1 = _mm256_loadu_si256((const __m256i*)(r0_ptr + 32));
                __m256i r0_v2 = _mm256_loadu_si256((const __m256i*)(r0_ptr + 64));
                __m256i r0_v3 = _mm256_loadu_si256((const __m256i*)(r0_ptr + 96));

                __m256i r1_v0 = _mm256_loadu_si256((const __m256i*)(r1_ptr));
                __m256i r1_v1 = _mm256_loadu_si256((const __m256i*)(r1_ptr + 32));
                __m256i r1_v2 = _mm256_loadu_si256((const __m256i*)(r1_ptr + 64));
                __m256i r1_v3 = _mm256_loadu_si256((const __m256i*)(r1_ptr + 96));

                auto calc_dense_16 = [&](__m256i v, __m256i coeffs) {
                    __m256i raw = compute_component_avx2(_mm256_unpacklo_epi8(v, zero), _mm256_unpackhi_epi8(v, zero), coeffs);
                    return _mm256_permute4x64_epi64(raw, 0xD8); 
                };

                __m256i y0_a = calc_dense_16(r0_v0, cY); // Pix 0-7 (Stored in Low 128)
                __m256i y0_b = calc_dense_16(r0_v1, cY); // Pix 8-15
                __m256i y0_c = calc_dense_16(r0_v2, cY); // Pix 16-23
                __m256i y0_d = calc_dense_16(r0_v3, cY); // Pix 24-31

                // Pack 16 -> 8
                __m256i y0_L = _mm256_packus_epi16(y0_a, y0_b); // Lane0: Pix 0-15
                __m256i y0_H = _mm256_packus_epi16(y0_c, y0_d); // Lane0: Pix 16-31

                // Combine Low 128 of both results
                // 0x20: Select Src1(Low) and Src2(Low)
                __m256i y0_final = _mm256_permute2x128_si256(y0_L, y0_H, 0x20);
                
                _mm256_storeu_si256((__m256i*)(dst.Y + y * width + x), y0_final);

                __m256i y1_a = calc_dense_16(r1_v0, cY);
                __m256i y1_b = calc_dense_16(r1_v1, cY);
                __m256i y1_c = calc_dense_16(r1_v2, cY);
                __m256i y1_d = calc_dense_16(r1_v3, cY);

                __m256i y1_L = _mm256_packus_epi16(y1_a, y1_b);
                __m256i y1_H = _mm256_packus_epi16(y1_c, y1_d);
                __m256i y1_final = _mm256_permute2x128_si256(y1_L, y1_H, 0x20);

                _mm256_storeu_si256((__m256i*)(dst.Y + (y + 1) * width + x), y1_final);

                __m256i avg0 = _mm256_avg_epu8(r0_v0, r1_v0);
                __m256i avg1 = _mm256_avg_epu8(r0_v1, r1_v1);
                __m256i avg2 = _mm256_avg_epu8(r0_v2, r1_v2);
                __m256i avg3 = _mm256_avg_epu8(r0_v3, r1_v3);

                // Calc UV (Dense & Offset)
                auto calc_uv_dense = [&](__m256i v, __m256i coeffs) {
                   __m256i res = calc_dense_16(v, coeffs);
                   return _mm256_add_epi16(res, offset128); // Add 128
                };

                __m256i u0 = calc_uv_dense(avg0, cU); // U0..7 (Low 128)
                __m256i u1 = calc_uv_dense(avg1, cU); // U8..15
                __m256i u2 = calc_uv_dense(avg2, cU); // U16..23
                __m256i u3 = calc_uv_dense(avg3, cU); // U24..31

                __m256i v0 = calc_uv_dense(avg0, cV);
                __m256i v1 = calc_uv_dense(avg1, cV);
                __m256i v2 = calc_uv_dense(avg2, cV);
                __m256i v3 = calc_uv_dense(avg3, cV);

                // Horizontal Downsample (HADD)
                // 1. Combine to Full Registers for HADD
                // U_0_15: [Lane0: U0..7 | Lane1: U8..15]
                // 0x20 = Lo(u0) | Lo(u1)
                __m256i U_0_15 = _mm256_permute2x128_si256(u0, u1, 0x20);
                __m256i U_16_31 = _mm256_permute2x128_si256(u2, u3, 0x20);

                __m256i V_0_15 = _mm256_permute2x128_si256(v0, v1, 0x20);
                __m256i V_16_31 = _mm256_permute2x128_si256(v2, v3, 0x20);

                // 2. Prepare for HADD (Shuffle High 64-bit of lanes to Low 64-bit of temp)
                // hadd inputs: [A, B]. Output: [Lo(A)+Hi(A)? No. Lo(A)+Lo(B)]
                // Correct logic for hadd_epi16:
                // Dest[0-63] = Src1[0-63] pairs added
                // Dest[64-127] = Src2[0-63] pairs added
                // So we need: Src1 = U, Src2 = Swapped(U)
                // 0x4E = Swap 64-bit blocks inside 128-bit lane
                __m256i U_0_15_swap = _mm256_shuffle_epi32(U_0_15, 0x4E);
                __m256i U_16_31_swap = _mm256_shuffle_epi32(U_16_31, 0x4E);
                
                __m256i V_0_15_swap = _mm256_shuffle_epi32(V_0_15, 0x4E);
                __m256i V_16_31_swap = _mm256_shuffle_epi32(V_16_31, 0x4E);

                // 3. HADD & Average
                // hadd(A, A_swap) -> Lane0: [Lo(A) sum, Lo(A_swap) sum] -> [0+1, 2+3, 4+5, 6+7]
                __m256i u_sum0 = _mm256_hadd_epi16(U_0_15, U_0_15_swap); 
                __m256i u_sum1 = _mm256_hadd_epi16(U_16_31, U_16_31_swap);
                __m256i v_sum0 = _mm256_hadd_epi16(V_0_15, V_0_15_swap);
                __m256i v_sum1 = _mm256_hadd_epi16(V_16_31, V_16_31_swap);

                u_sum0 = _mm256_srai_epi16(u_sum0, 1);
                u_sum1 = _mm256_srai_epi16(u_sum1, 1);
                v_sum0 = _mm256_srai_epi16(v_sum0, 1);
                v_sum1 = _mm256_srai_epi16(v_sum1, 1);

                // 4. Pack & Store
                // u_sum0 Lane0: U0..7 (Correctly ordered now). Lane1: U8..15 (Correctly ordered).
                // u_sum1 Lane0: U16..23. Lane1: U24..31.
                
                // Packus packs Lane0(A) and Lane0(B) -> Dest Lane 0.
                // Lane 0: Pack(U0..7, U16..23) -> Order: 0..7, 16..23
                // Lane 1: Pack(U8..15, U24..31) -> Order: 8..15, 24..31
                __m256i u_final = _mm256_packus_epi16(u_sum0, u_sum1);
                __m256i v_final = _mm256_packus_epi16(v_sum0, v_sum1);

                // Fix Order: 0, 2, 1, 3 (0xD8)
                // Res: 0..7, 8..15, 16..23, 24..31
                u_final = _mm256_permute4x64_epi64(u_final, 0xD8);
                v_final = _mm256_permute4x64_epi64(v_final, 0xD8);

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                _mm_storeu_si128((__m128i*)(dst.U + uvIdx), _mm256_castsi256_si128(u_final));
                _mm_storeu_si128((__m128i*)(dst.V + uvIdx), _mm256_castsi256_si128(v_final));
            }
        }
        _mm256_zeroupper();
    }

    // =========================================================
    // RGB888 -> YUV420 (AVX2 Fixed)
    // =========================================================
    void RGB2YUV_RGB888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        __m256i cY = _mm256_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0, 29, 150, 77, 0, 29, 150, 77, 0);
        __m256i cU = _mm256_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0);
        __m256i cV = _mm256_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0);

        __m256i zero = _mm256_setzero_si256();
        __m256i offset128 = _mm256_set1_epi16(128);

        // RGB -> BGRA Expand Mask
        __m128i rgb_expand_mask = _mm_setr_epi8(
            0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1
        );

        auto load_rgb_as_bgra_8px = [&](const uint8_t* p) -> __m256i {
            __m128i raw_lo = _mm_loadu_si128((const __m128i*)p);
            __m128i bgra_lo = _mm_shuffle_epi8(raw_lo, rgb_expand_mask);
            __m128i raw_hi = _mm_loadu_si128((const __m128i*)(p + 12));
            __m128i bgra_hi = _mm_shuffle_epi8(raw_hi, rgb_expand_mask);
            return _mm256_inserti128_si256(_mm256_castsi128_si256(bgra_lo), bgra_hi, 1);
        };

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 32)
            {
                const uint8_t* r0_ptr = src.data + (y * width + x) * 3;
                const uint8_t* r1_ptr = src.data + ((y + 1) * width + x) * 3;

                // Load Data
                __m256i r0_v0 = load_rgb_as_bgra_8px(r0_ptr);
                __m256i r0_v1 = load_rgb_as_bgra_8px(r0_ptr + 24);
                __m256i r0_v2 = load_rgb_as_bgra_8px(r0_ptr + 48);
                __m256i r0_v3 = load_rgb_as_bgra_8px(r0_ptr + 72);

                __m256i r1_v0 = load_rgb_as_bgra_8px(r1_ptr);
                __m256i r1_v1 = load_rgb_as_bgra_8px(r1_ptr + 24);
                __m256i r1_v2 = load_rgb_as_bgra_8px(r1_ptr + 48);
                __m256i r1_v3 = load_rgb_as_bgra_8px(r1_ptr + 72);

                 auto calc_dense_16 = [&](__m256i v, __m256i coeffs) {
                    __m256i raw = compute_component_avx2(_mm256_unpacklo_epi8(v, zero), _mm256_unpackhi_epi8(v, zero), coeffs);
                    return _mm256_permute4x64_epi64(raw, 0xD8); 
                };

                // Y Row 0
                __m256i y0_a = calc_dense_16(r0_v0, cY);
                __m256i y0_b = calc_dense_16(r0_v1, cY);
                __m256i y0_c = calc_dense_16(r0_v2, cY);
                __m256i y0_d = calc_dense_16(r0_v3, cY);

                __m256i y0_L = _mm256_packus_epi16(y0_a, y0_b);
                __m256i y0_H = _mm256_packus_epi16(y0_c, y0_d);
                __m256i y0_final = _mm256_permute2x128_si256(y0_L, y0_H, 0x20);
                _mm256_storeu_si256((__m256i*)(dst.Y + y * width + x), y0_final);

                // Y Row 1
                __m256i y1_a = calc_dense_16(r1_v0, cY);
                __m256i y1_b = calc_dense_16(r1_v1, cY);
                __m256i y1_c = calc_dense_16(r1_v2, cY);
                __m256i y1_d = calc_dense_16(r1_v3, cY);

                __m256i y1_L = _mm256_packus_epi16(y1_a, y1_b);
                __m256i y1_H = _mm256_packus_epi16(y1_c, y1_d);
                __m256i y1_final = _mm256_permute2x128_si256(y1_L, y1_H, 0x20);
                _mm256_storeu_si256((__m256i*)(dst.Y + (y + 1) * width + x), y1_final);

                // UV
                __m256i avg0 = _mm256_avg_epu8(r0_v0, r1_v0);
                __m256i avg1 = _mm256_avg_epu8(r0_v1, r1_v1);
                __m256i avg2 = _mm256_avg_epu8(r0_v2, r1_v2);
                __m256i avg3 = _mm256_avg_epu8(r0_v3, r1_v3);

                auto calc_uv_dense = [&](__m256i v, __m256i coeffs) {
                   __m256i res = calc_dense_16(v, coeffs);
                   return _mm256_add_epi16(res, offset128);
                };

                __m256i u0 = calc_uv_dense(avg0, cU);
                __m256i u1 = calc_uv_dense(avg1, cU);
                __m256i u2 = calc_uv_dense(avg2, cU);
                __m256i u3 = calc_uv_dense(avg3, cU);

                __m256i v0 = calc_uv_dense(avg0, cV);
                __m256i v1 = calc_uv_dense(avg1, cV);
                __m256i v2 = calc_uv_dense(avg2, cV);
                __m256i v3 = calc_uv_dense(avg3, cV);

                __m256i U_0_15 = _mm256_permute2x128_si256(u0, u1, 0x20);
                __m256i U_16_31 = _mm256_permute2x128_si256(u2, u3, 0x20);
                __m256i V_0_15 = _mm256_permute2x128_si256(v0, v1, 0x20);
                __m256i V_16_31 = _mm256_permute2x128_si256(v2, v3, 0x20);

                __m256i U_0_15_swap = _mm256_shuffle_epi32(U_0_15, 0x4E);
                __m256i U_16_31_swap = _mm256_shuffle_epi32(U_16_31, 0x4E);
                __m256i V_0_15_swap = _mm256_shuffle_epi32(V_0_15, 0x4E);
                __m256i V_16_31_swap = _mm256_shuffle_epi32(V_16_31, 0x4E);

                __m256i u_sum0 = _mm256_hadd_epi16(U_0_15, U_0_15_swap); 
                __m256i u_sum1 = _mm256_hadd_epi16(U_16_31, U_16_31_swap);
                __m256i v_sum0 = _mm256_hadd_epi16(V_0_15, V_0_15_swap);
                __m256i v_sum1 = _mm256_hadd_epi16(V_16_31, V_16_31_swap);

                u_sum0 = _mm256_srai_epi16(u_sum0, 1);
                u_sum1 = _mm256_srai_epi16(u_sum1, 1);
                v_sum0 = _mm256_srai_epi16(v_sum0, 1);
                v_sum1 = _mm256_srai_epi16(v_sum1, 1);

                __m256i u_final = _mm256_packus_epi16(u_sum0, u_sum1);
                __m256i v_final = _mm256_packus_epi16(v_sum0, v_sum1);

                u_final = _mm256_permute4x64_epi64(u_final, 0xD8);
                v_final = _mm256_permute4x64_epi64(v_final, 0xD8);

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                _mm_storeu_si128((__m128i*)(dst.U + uvIdx), _mm256_castsi256_si128(u_final));
                _mm_storeu_si128((__m128i*)(dst.V + uvIdx), _mm256_castsi256_si128(v_final));
            }
        }
        _mm256_zeroupper();
    }

    // ---------------------------------------------------------
    // MemOnly: 模拟 Shuffle 后的 12字节 (8+4) 块写入
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_MemOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        uint8_t *dstPtr = dst.data;
        const uint8_t *YPtr = src.Y;

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            // Force read UV
            volatile __m128i u_v = _mm_loadu_si128((const __m128i*)(src.U + uvOffset));
            (void)u_v;

            for (int x = 0; x < width; x += 32)
            {
                // 1. Load Y
                __m256i y_raw = _mm256_loadu_si256((const __m256i *)(YPtr + y * width + x));

                // 2. Dummy Compute Result (模拟四个 Lane 的结果)
                // 原版代码中，s0, s1, s2, s3 已经是 Shuffle 好的数据
                __m256i s_dummy = y_raw; 

                uint8_t* p = dstPtr + (y * width + x) * 3;

                // 3. Store Lambda (复刻原版)
                auto store_lane_12bytes = [&](__m128i lane_data) {
                    _mm_storel_epi64((__m128i*)p, lane_data);
                    *(int*)(p + 8) = _mm_cvtsi128_si32(_mm_srli_si128(lane_data, 8));
                    p += 12; 
                };

                // 执行存储模式
                // Lane 0 & 1
                store_lane_12bytes(_mm256_castsi256_si128(s_dummy));
                store_lane_12bytes(_mm256_extracti128_si256(s_dummy, 1));
                
                // 模拟后续 Lane (虽然数据一样，但 Store 指令流是一样的)
                store_lane_12bytes(_mm256_castsi256_si128(s_dummy));
                store_lane_12bytes(_mm256_extracti128_si256(s_dummy, 1));

                store_lane_12bytes(_mm256_castsi256_si128(s_dummy));
                store_lane_12bytes(_mm256_extracti128_si256(s_dummy, 1));

                store_lane_12bytes(_mm256_castsi256_si128(s_dummy));
                store_lane_12bytes(_mm256_extracti128_si256(s_dummy, 1));
            }
        }
    }

    // ---------------------------------------------------------
    // ComputeOnly: 包含所有 Permute 和 Shuffle
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_ComputeOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;

        __m256i c90  = _mm256_set1_epi16(90);
        __m256i c22  = _mm256_set1_epi16(22);
        __m256i c46  = _mm256_set1_epi16(46);
        __m256i c113 = _mm256_set1_epi16(113);
        __m256i c128 = _mm256_set1_epi16(128);
        __m256i zero = _mm256_setzero_si256();
        __m256i alphaVec = zero; 
        
        // Critical Shuffle Mask
        __m256i shuffleMask = _mm256_setr_epi8(
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1 
        );

        __m256i accum = zero;

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 32)
            {
                // ... Load & Upsample ...
                __m256i y_raw = _mm256_loadu_si256((const __m256i *)(YPtr + y * width + x));
                __m128i u_small = _mm_loadu_si128((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_small = _mm_loadu_si128((const __m128i *)(VPtr + uvOffset + (x / 2)));

                __m128i u_lo_128 = _mm_unpacklo_epi8(u_small, u_small); 
                __m128i u_hi_128 = _mm_unpackhi_epi8(u_small, u_small);
                __m256i u_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(u_lo_128), u_hi_128, 1);

                __m128i v_lo_128 = _mm_unpacklo_epi8(v_small, v_small);
                __m128i v_hi_128 = _mm_unpackhi_epi8(v_small, v_small);
                __m256i v_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(v_lo_128), v_hi_128, 1);

                // ... Calc ...
                __m256i y0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_raw));
                __m256i u0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_raw)), c128);
                __m256i v0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_raw)), c128);

                __m256i y1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_raw, 1));
                __m256i u1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(u_raw, 1)), c128);
                __m256i v1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(v_raw, 1)), c128);

                __m256i r0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v0), 6));
                __m256i g_part0 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u0), _mm256_mullo_epi16(c46, v0));
                __m256i g0 = _mm256_sub_epi16(y0, _mm256_srai_epi16(g_part0, 6));
                __m256i b0 = _mm256_add_epi16(y0, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u0), 6));

                __m256i r1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c90, v1), 6));
                __m256i g_part1 = _mm256_add_epi16(_mm256_mullo_epi16(c22, u1), _mm256_mullo_epi16(c46, v1));
                __m256i g1 = _mm256_sub_epi16(y1, _mm256_srai_epi16(g_part1, 6));
                __m256i b1 = _mm256_add_epi16(y1, _mm256_srai_epi16(_mm256_mullo_epi16(c113, u1), 6));

                // ... Pack & Permute ...
                __m256i R = _mm256_packus_epi16(r0, r1);
                __m256i G = _mm256_packus_epi16(g0, g1);
                __m256i B = _mm256_packus_epi16(b0, b1);
                __m256i A = alphaVec;

                R = _mm256_permute4x64_epi64(R, 0xD8);
                G = _mm256_permute4x64_epi64(G, 0xD8);
                B = _mm256_permute4x64_epi64(B, 0xD8);

                __m256i bg_lo = _mm256_unpacklo_epi8(B, G);
                __m256i bg_hi = _mm256_unpackhi_epi8(B, G);
                __m256i ra_lo = _mm256_unpacklo_epi8(R, A);
                __m256i ra_hi = _mm256_unpackhi_epi8(R, A);

                __m256i res0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                __m256i res1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);
                __m256i res2 = _mm256_unpacklo_epi16(bg_hi, ra_hi);
                __m256i res3 = _mm256_unpackhi_epi16(bg_hi, ra_hi);

                __m256i out0 = _mm256_permute2x128_si256(res0, res1, 0x20);
                __m256i out1 = _mm256_permute2x128_si256(res2, res3, 0x20);
                __m256i out2 = _mm256_permute2x128_si256(res0, res1, 0x31);
                __m256i out3 = _mm256_permute2x128_si256(res2, res3, 0x31);

                // ... Final Shuffle (Important part of compute) ...
                __m256i s0 = _mm256_shuffle_epi8(out0, shuffleMask);
                __m256i s1 = _mm256_shuffle_epi8(out1, shuffleMask);
                __m256i s2 = _mm256_shuffle_epi8(out2, shuffleMask);
                __m256i s3 = _mm256_shuffle_epi8(out3, shuffleMask);

                // 虚假累加
                accum = _mm256_xor_si256(accum, s0);
                accum = _mm256_xor_si256(accum, s1);
                accum = _mm256_xor_si256(accum, s2);
                accum = _mm256_xor_si256(accum, s3);
            }
        }
        volatile int keep_alive = _mm256_extract_epi32(accum, 0);
        (void)keep_alive;
        _mm256_zeroupper();
    }

    // =========================================================
    // Analysis: Shuffle/Layout Overhead Test
    // 目的：测量数据重排（Permute/Shuffle/Pack）的开销
    // =========================================================
    void YUV2RGB_RGB888_ShuffleOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;

        __m256i zero = _mm256_setzero_si256();
        // 只有 Shuffle Mask 是必须的，数学常数都不需要了
        __m256i shuffleMask = _mm256_setr_epi8(
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1 
        );

        __m256i accum = zero;

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            for (int x = 0; x < width; x += 32)
            {
                // 1. Load & Upsample (属于 Layout 操作)
                __m256i y_raw = _mm256_loadu_si256((const __m256i *)(YPtr + y * width + x));
                
                __m128i u_small = _mm_loadu_si128((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_small = _mm_loadu_si128((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // UV Upsample (Unpack + Insert)
                __m128i u_lo_128 = _mm_unpacklo_epi8(u_small, u_small); 
                __m128i u_hi_128 = _mm_unpackhi_epi8(u_small, u_small);
                __m256i u_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(u_lo_128), u_hi_128, 1);

                __m128i v_lo_128 = _mm_unpacklo_epi8(v_small, v_small);
                __m128i v_hi_128 = _mm_unpackhi_epi8(v_small, v_small);
                __m256i v_raw = _mm256_inserti128_si256(_mm256_castsi128_si256(v_lo_128), v_hi_128, 1);

                // 2. Math Removed (跳过 Convert/Sub/Mul/Add)
                // 直接把 Y/U/V 当作计算结果，模拟数据流
                // 为了模拟数据依赖链，我们做简单的 XOR 混合，延迟极低
                __m256i r0 = y_raw;
                __m256i g0 = _mm256_xor_si256(y_raw, u_raw);
                __m256i b0 = v_raw;
                
                // 模拟两个 Batch 的结果 (r1/g1/b1)
                __m256i r1 = r0;
                __m256i g1 = g0;
                __m256i b1 = b0;

                // 3. Pack & Permute & Interleave (核心 Layout 开销)
                // Pack 16->8
                __m256i R = _mm256_packus_epi16(r0, r1);
                __m256i G = _mm256_packus_epi16(g0, g1);
                __m256i B = _mm256_packus_epi16(b0, b1);
                __m256i A = zero; // Alpha

                // Permute (跨 Lane 修正)
                R = _mm256_permute4x64_epi64(R, 0xD8);
                G = _mm256_permute4x64_epi64(G, 0xD8);
                B = _mm256_permute4x64_epi64(B, 0xD8);

                // Interleave (Unpack)
                __m256i bg_lo = _mm256_unpacklo_epi8(B, G);
                __m256i bg_hi = _mm256_unpackhi_epi8(B, G);
                __m256i ra_lo = _mm256_unpacklo_epi8(R, A);
                __m256i ra_hi = _mm256_unpackhi_epi8(R, A);

                __m256i res0 = _mm256_unpacklo_epi16(bg_lo, ra_lo);
                __m256i res1 = _mm256_unpackhi_epi16(bg_lo, ra_lo);
                __m256i res2 = _mm256_unpacklo_epi16(bg_hi, ra_hi);
                __m256i res3 = _mm256_unpackhi_epi16(bg_hi, ra_hi);

                // Permute2x128 (跨 128-bit Lane 重组)
                __m256i out0 = _mm256_permute2x128_si256(res0, res1, 0x20);
                __m256i out1 = _mm256_permute2x128_si256(res2, res3, 0x20);
                __m256i out2 = _mm256_permute2x128_si256(res0, res1, 0x31);
                __m256i out3 = _mm256_permute2x128_si256(res2, res3, 0x31);

                // Shuffle (Final Compression to RGB)
                __m256i s0 = _mm256_shuffle_epi8(out0, shuffleMask);
                __m256i s1 = _mm256_shuffle_epi8(out1, shuffleMask);
                __m256i s2 = _mm256_shuffle_epi8(out2, shuffleMask);
                __m256i s3 = _mm256_shuffle_epi8(out3, shuffleMask);

                // 4. Accumulate (No Store)
                accum = _mm256_xor_si256(accum, s0);
                accum = _mm256_xor_si256(accum, s1);
                accum = _mm256_xor_si256(accum, s2);
                accum = _mm256_xor_si256(accum, s3);
            }
        }
        volatile int keep = _mm256_extract_epi32(accum, 0); (void)keep;
    }
}