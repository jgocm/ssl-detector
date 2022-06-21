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

#include "tfliteNMSPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::TFLiteNMSBasePluginCreator;
using nvinfer1::plugin::TFLiteNMSPlugin;
using nvinfer1::plugin::TFLiteNMSPluginCreator;
using nvinfer1::plugin::TFLiteNMSParameters;

namespace
{
const char* TFLITE_NMS_PLUGIN_VERSION{"1"};
const char* TFLITE_NMS_PLUGIN_NAMES[] = {"TFLiteNMS_TRT"};
} // namespace

PluginFieldCollection TFLiteNMSBasePluginCreator::mFC{};
std::vector<PluginField> TFLiteNMSBasePluginCreator::mPluginAttributes;

TFLiteNMSPlugin::TFLiteNMSPlugin(TFLiteNMSParameters params) noexcept
    : param(params)
{
}

TFLiteNMSPlugin::TFLiteNMSPlugin(const void* data, size_t length) noexcept
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<TFLiteNMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    anchorsSize = read<int>(d);
    numPriors = read<int>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    ASSERT(d == a + length);
}

int TFLiteNMSPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int TFLiteNMSPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void TFLiteNMSPlugin::terminate() noexcept {}

Dims TFLiteNMSPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    ASSERT(nbInputDims == 3);
    ASSERT(index >= 0 && index < this->getNbOutputs());
    ASSERT(inputs[0].nbDims == 3 || inputs[0].nbDims == 2);
    ASSERT(inputs[1].nbDims == 2 || (inputs[1].nbDims == 3 && inputs[1].d[2] == 1));
    ASSERT(inputs[2].nbDims == 2);
    // boxesSize: number of box coordinates for one sample
    boxesSize = 1;
    for (auto i = 0; i < inputs[0].nbDims; i++)
    {
        boxesSize *= inputs[0].d[i];
    }
    // scoresSize: number of scores for one sample
    scoresSize = inputs[1].d[0] * inputs[1].d[1];
    // anchorSize: number of anchors for one sample
    anchorsSize = inputs[2].d[0] * inputs[2].d[1];

    // num_detections
    if (index == 0)
    {
        Dims dim0{};
        dim0.nbDims = 0;
        return dim0;
    }
    // nmsed_boxes
    if (index == 1)
    {
        return DimsHW(param.max_detections, 4);
    }
    // nmsed_scores or nmsed_classes
    Dims dim1{};
    dim1.nbDims = 1;
    dim1.d[0] = param.max_detections;
    return dim1;
}

size_t TFLiteNMSPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
   return tfliteDetectionInferenceWorkspaceSize(true, maxBatchSize, boxesSize, scoresSize, anchorsSize,
        param.num_classes, numPriors, numPriors, mPrecision, mPrecision, mPrecision);
}

int TFLiteNMSPlugin::enqueue(
    int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* const locData = inputs[0];
    const void* const confData = inputs[1];
    const void* const anchorData = inputs[2];

    void* keepCount = outputs[0];
    void* nmsedBoxes = outputs[1];
    void* nmsedScores = outputs[2];
    void* nmsedClasses = outputs[3];

    const int topK = numPriors;

    pluginStatus_t status = tfliteNMSInference(
        stream,
        batchSize,
        boxesSize,
        scoresSize,
        anchorsSize,
        numPriors,
        param.num_classes,
        param.max_detections,
        param.back_ground_Label_id,
        param.nms_score_threshold,
        param.nms_iou_threshold,
        param.y_scale,
        param.x_scale,
        param.h_scale,
        param.w_scale,
        mPrecision,
        locData,
        mPrecision,
        confData,
        mPrecision,
        anchorData,
        keepCount,
        nmsedBoxes,
        nmsedScores,
        nmsedClasses,
        workspace,
        false,
        mClipBoxes,
        mScoreBits);
    ASSERT(status == STATUS_SUCCESS);
    return 0;
}

size_t TFLiteNMSPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, boxesSize,scoresSize,anchorsSize,numPriors
    return sizeof(TFLiteNMSParameters) + sizeof(int) * 4 + sizeof(DataType) + sizeof(int32_t);
}

void TFLiteNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, anchorsSize);
    write(d, numPriors);
    write(d, mPrecision);
    write(d, mScoreBits);
    ASSERT(d == a + getSerializationSize());
}

void TFLiteNMSPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    ASSERT(nbInputs == 3);
    ASSERT(nbOutputs == 4);
    ASSERT(inputDims[0].nbDims == 3 || inputDims[0].nbDims == 2);
    ASSERT(inputDims[1].nbDims == 2 || (inputDims[1].nbDims == 3 && inputDims[1].d[2] == 1));
    ASSERT(inputDims[2].nbDims == 2);
    ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs, [](bool b) { return b; }));
    ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs, [](bool b) { return b; }));

    boxesSize = 1;
    for (auto i = 0; i < inputDims[0].nbDims; i++)
    {
        std::cout << i << ": " << inputDims[0].d[i] << std::endl;
        boxesSize *= inputDims[0].d[i];
    }
    scoresSize = inputDims[1].d[0] * inputDims[1].d[1];
    anchorsSize = inputDims[2].d[0] * inputDims[2].d[1];
    // num_boxes
    numPriors = inputDims[0].d[0];
    const int numLocClasses = 1;
    // Third dimension of boxes must be either 1 or num_classes
    if (inputDims[0].nbDims == 3)
    {
        ASSERT(inputDims[0].d[1] == numLocClasses);
        ASSERT(inputDims[0].d[2] == 4);
    }
    else
    {
        ASSERT(inputDims[0].nbDims == 2 && inputDims[0].d[1] == 4);
    }
    mPrecision = inputTypes[0];
}

bool TFLiteNMSPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32)
        && format == PluginFormat::kLINEAR);
}

const char* TFLiteNMSPlugin::getPluginType() const noexcept
{
    return TFLITE_NMS_PLUGIN_NAMES[0];
}

const char* TFLiteNMSPlugin::getPluginVersion() const noexcept
{
    return TFLITE_NMS_PLUGIN_VERSION;
}

void TFLiteNMSPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* TFLiteNMSPlugin::clone() const noexcept
{
    auto* plugin = new TFLiteNMSPlugin(param);
    plugin->boxesSize = boxesSize;
    plugin->scoresSize = scoresSize;
    plugin->numPriors = numPriors;
    plugin->anchorsSize = anchorsSize;
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->mPrecision = mPrecision;
    plugin->setScoreBits(mScoreBits);
    return plugin;
}

void TFLiteNMSPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* TFLiteNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType TFLiteNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void TFLiteNMSPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

bool TFLiteNMSPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    noexcept
{
    return false;
}

bool TFLiteNMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

TFLiteNMSBasePluginCreator::TFLiteNMSBasePluginCreator() noexcept
    : params{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("maxClassesPerDetection", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("yScale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("xScale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("hScale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("wScale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreBits", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

TFLiteNMSPluginCreator::TFLiteNMSPluginCreator() noexcept
{
    mPluginName = TFLITE_NMS_PLUGIN_NAMES[0];
}

const char* TFLiteNMSBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* TFLiteNMSBasePluginCreator::getPluginVersion() const noexcept
{
    return TFLITE_NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* TFLiteNMSBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* TFLiteNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    mScoreBits = 16;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "maxClassesPerDetection"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.max_classes_per_detection = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.max_detections = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.back_ground_Label_id = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "iouThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nms_iou_threshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scoreThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nms_score_threshold = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.num_classes = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "yScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.y_scale = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "xScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.x_scale = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "hScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.h_scale = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "wScale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.w_scale = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attrName, "scoreBits"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mScoreBits = *(static_cast<const int32_t*>(fields[i].data));
        }
    }

    auto* plugin = new TFLiteNMSPlugin(params);
    plugin->setScoreBits(mScoreBits);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* TFLiteNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call NMS::destroy()
    auto* plugin = new TFLiteNMSPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
