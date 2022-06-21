/*
 * Copyright (c) 2021 Nobuo Tsukamoto
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

#ifndef TFLITE_NMS_PLUGIN_H
#define TFLITE_NMS_PLUGIN_H
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

struct TFLiteNMSParameters
{
    int32_t max_classes_per_detection;
    int32_t max_detections;
    int32_t back_ground_Label_id;
    float nms_iou_threshold;
    float nms_score_threshold;
    float num_classes;
    float y_scale;
    float x_scale;
    float h_scale;
    float w_scale;
};

class TFLiteNMSPlugin : public IPluginV2Ext
{
public:
    TFLiteNMSPlugin(TFLiteNMSParameters param) noexcept;
    TFLiteNMSPlugin(const void* data, size_t length) noexcept;
    ~TFLiteNMSPlugin() noexcept override = default;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void setScoreBits(int32_t scoreBits) noexcept;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const
        noexcept override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
        noexcept override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;
    IPluginV2Ext* clone() const noexcept override;

private:
    TFLiteNMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int anchorsSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes = false;
    DataType mPrecision;
    int32_t mScoreBits;
};

class TFLiteNMSBasePluginCreator : public BaseCreator
{
public:
    TFLiteNMSBasePluginCreator() noexcept;
    ~TFLiteNMSBasePluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

protected:
    static PluginFieldCollection mFC;
    TFLiteNMSParameters params;
    static std::vector<PluginField> mPluginAttributes;
    int32_t mScoreBits;
    std::string mPluginName;
};

class TFLiteNMSPluginCreator : public TFLiteNMSBasePluginCreator
{
public:
    TFLiteNMSPluginCreator() noexcept;
    ~TFLiteNMSPluginCreator() noexcept override = default;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TFLITE_NMS_PLUGIN_H
