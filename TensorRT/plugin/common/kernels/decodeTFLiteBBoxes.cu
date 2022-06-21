/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <array>
#include "kernel.h"
#include "cuda_fp16.h"

// overloading exp for half type
inline __device__ __half exp(__half a) {
#if __CUDA_ARCH__ >= 530
    return hexp(a);
#else
    return exp(float(a));
#endif
}

inline __device__ __half add_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a + b;
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
}

inline __device__ __half minus_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a - b;
#else
    return __float2half(__half2float(a) - __half2float(b));
#endif
}

inline __device__ __half mul_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a * b;
#else
    return __float2half(__half2float(a) * __half2float(b));
#endif
}

inline __device__ __half div_fb(const __half & a, const __half & b) {
#if __CUDA_ARCH__ >= 530
    return a / b;
#else
    return __float2half(__half2float(a) / __half2float(b));
#endif
}

template <typename T_BBOX, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void decodeTFLiteBBoxes_kernel(
        const int nthreads,
        const int num_priors,
        const int num_loc_classes,
        const T_BBOX* loc_data,
        const T_BBOX* prior_data,
        const float scaleY,
        const float scaleX,
        const float scaleH,
        const float scaleW,
        T_BBOX* bbox_data)
{
    // nthds_per_cta = 512
    for (int index = blockIdx.x * nthds_per_cta + threadIdx.x;
         index < nthreads;
         index += nthds_per_cta * gridDim.x)
    {
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc

        // Bounding box coordinate index {0, 1, 2, 3}
        const int box_encoding_idx = index * 4;

        // Anchor
        const T_BBOX anchorY = prior_data[box_encoding_idx];
        const T_BBOX anchorX = prior_data[box_encoding_idx + 1];
        const T_BBOX anchorH = prior_data[box_encoding_idx + 2];
        const T_BBOX anchorW = prior_data[box_encoding_idx + 3];

        // BBox
        const T_BBOX boxY = loc_data[box_encoding_idx];
        const T_BBOX boxX = loc_data[box_encoding_idx + 1];
        const T_BBOX boxH = loc_data[box_encoding_idx + 2];
        const T_BBOX boxW = loc_data[box_encoding_idx + 3];

        // ycenter = y / y_scale * anchor.h + anchor.y;
        // xcenter = x / x_scale * anchor.w + anchor.x;
        // half_h = 0.5*exp(h / h_scale)) * anchor.h;
        // half_w = 0.5*exp(w / w_scale)) * anchor.w;
        T_BBOX bboxCenterY = add_fb(mul_fb(div_fb(boxY, T_BBOX(scaleY)), anchorH), anchorY);
        T_BBOX bboxCenterX = add_fb(mul_fb(div_fb(boxX, T_BBOX(scaleX)), anchorW), anchorX);
        T_BBOX harfH = div_fb(mul_fb(exp(div_fb(boxH, T_BBOX(scaleH))), anchorH), T_BBOX(2));
        T_BBOX harfW = div_fb(mul_fb(exp(div_fb(boxW, T_BBOX(scaleW))), anchorW), T_BBOX(2));

        // xmin = xcenter - half_w
        // ymin = ycenter - half_h
        // xmax = xcenter + half_w
        // ymax = ycenter + half_h
        bbox_data[box_encoding_idx] = minus_fb(bboxCenterX, harfW);
        bbox_data[box_encoding_idx + 1] = minus_fb(bboxCenterY, harfH);
        bbox_data[box_encoding_idx + 2] = add_fb(bboxCenterX, harfW);
        bbox_data[box_encoding_idx + 3] = add_fb(bboxCenterY, harfH);

        // Forcibly restrict bounding boxes to the normalized range [0,1].
        bbox_data[box_encoding_idx] = saturate(bbox_data[box_encoding_idx]);
        bbox_data[box_encoding_idx + 1] = saturate(bbox_data[box_encoding_idx + 1]);
        bbox_data[box_encoding_idx + 2] = saturate(bbox_data[box_encoding_idx + 2]);
        bbox_data[box_encoding_idx + 3] = saturate(bbox_data[box_encoding_idx + 3]);
    }
}

template <typename T_BBOX>
pluginStatus_t decodeTFLiteBBoxes_gpu(
    cudaStream_t stream,
    const int nthreads,
    const int num_priors,
    const int num_loc_classes,
    const void* loc_data,
    const void* prior_data,
    const float scaleY,
    const float scaleX,
    const float scaleH,
    const float scaleW,
    void* bbox_data)
{
    const int BS = 512;
    const int GS = (nthreads + BS - 1) / BS;
    decodeTFLiteBBoxes_kernel<T_BBOX, BS><<<GS, BS, 0, stream>>>(nthreads,
                                                                 num_priors,
                                                                 num_loc_classes,
                                                                 (const T_BBOX*) loc_data,
                                                                 (const T_BBOX*) prior_data,
                                                                 scaleY,
                                                                 scaleX,
                                                                 scaleH,
                                                                 scaleW,
                                                                 (T_BBOX*) bbox_data);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// decodeTFLiteBBoxes LAUNCH CONFIG
typedef pluginStatus_t (*dbbTFLiteFunc)(cudaStream_t,
                                        const int,
                                        const int,
                                        const int,
                                        const void*,
                                        const void*,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        void*);

struct decodeTFLiteBBoxesLaunchConfig
{
    DataType t_bbox;
    dbbTFLiteFunc function;

    decodeTFLiteBBoxesLaunchConfig(DataType t_bbox)
        : t_bbox(t_bbox)
    {
    }
    decodeTFLiteBBoxesLaunchConfig(DataType t_bbox, dbbTFLiteFunc function)
        : t_bbox(t_bbox)
        , function(function)
    {
    }
    bool operator==(const decodeTFLiteBBoxesLaunchConfig& other)
    {
        return t_bbox == other.t_bbox;
    }
};

static std::array<decodeTFLiteBBoxesLaunchConfig, 2> decodeTFLiteBBoxesLCOptions = {
    decodeTFLiteBBoxesLaunchConfig(DataType::kFLOAT, decodeTFLiteBBoxes_gpu<float>),
    decodeTFLiteBBoxesLaunchConfig(DataType::kHALF, decodeTFLiteBBoxes_gpu<__half>)
};

pluginStatus_t decodeTFLiteBBoxes(
    cudaStream_t stream,
    const int nthreads,
    const int num_priors,
    const int num_loc_classes,
    const DataType DT_BBOX,
    const void* loc_data,
    const void* prior_data,
    const float scaleY,
    const float scaleX,
    const float scaleH,
    const float scaleW,
    void* bbox_data)
{
    decodeTFLiteBBoxesLaunchConfig lc = decodeTFLiteBBoxesLaunchConfig(DT_BBOX);
    for (unsigned i = 0; i < decodeTFLiteBBoxesLCOptions.size(); ++i)
    {
        if (lc == decodeTFLiteBBoxesLCOptions[i])
        {
            DEBUG_PRINTF("decodeTFLiteBBox kernel %d\n", i);
            return decodeTFLiteBBoxesLCOptions[i].function(stream,
                                            nthreads,
                                            num_priors,
                                            num_loc_classes,
                                            loc_data,
                                            prior_data,
                                            scaleY,
                                            scaleX,
                                            scaleH,
                                            scaleW,
                                            bbox_data);
        }
    }
    return STATUS_BAD_PARAM;
}
