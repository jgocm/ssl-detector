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

#include "common.h"
#include "lstmDecoder.h"
#include "trtUtil.h"
#include <fstream>
#include <sstream>

namespace nmtSample
{
LSTMDecoder::LSTMDecoder(ComponentWeights::ptr& weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    ASSERT(mWeights->mMetaData.size() >= 4);
    nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    ASSERT(dataType == nvinfer1::DataType::kFLOAT);
    mRNNKind = mWeights->mMetaData[1];
    mNumLayers = mWeights->mMetaData[2];
    mNumUnits = mWeights->mMetaData[3];
    size_t elementSize = inferTypeToBytes(dataType);
    // compute weights offsets
    size_t dataSize = 2 * mNumUnits;
    size_t kernelOffset = 0;
    size_t biasStartOffset = ((4 * dataSize + 4 * mNumUnits) * mNumUnits) * elementSize
        + 8 * mNumUnits * mNumUnits * (mNumLayers - 1) * elementSize;
    size_t biasOffset = biasStartOffset;
    int32_t numGates = 8;
    for (int32_t layerIndex = 0; layerIndex < mNumLayers; layerIndex++)
    {
        for (int32_t gateIndex = 0; gateIndex < numGates; gateIndex++)
        {
            // encoder input size == mNumUnits
            int64_t inputSize = ((layerIndex == 0) && (gateIndex < 4)) ? dataSize : mNumUnits;
            nvinfer1::Weights gateKernelWeights{dataType, &mWeights->mWeights[0] + kernelOffset, inputSize * mNumUnits};
            nvinfer1::Weights gateBiasWeights{dataType, &mWeights->mWeights[0] + biasOffset, mNumUnits};
            mGateKernelWeights.push_back(std::move(gateKernelWeights));
            mGateBiasWeights.push_back(std::move(gateBiasWeights));
            kernelOffset = kernelOffset + inputSize * mNumUnits * elementSize;
            biasOffset = biasOffset + mNumUnits * elementSize;
        }
    }
    ASSERT(kernelOffset + biasOffset - biasStartOffset == mWeights->mWeights.size());
}

void LSTMDecoder::addToModel(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* inputEmbeddedData,
    nvinfer1::ITensor** inputStates, nvinfer1::ITensor** outputData, nvinfer1::ITensor** outputStates)
{
    int32_t beamWidth;
    int32_t inputWidth;
    {
        auto dims = inputEmbeddedData->getDimensions();
        ASSERT(dims.nbDims == 2);
        beamWidth = dims.d[0];
        inputWidth = dims.d[1];
    }

    nvinfer1::ITensor* shuffledInput;
    {
        auto shuffleLayer = network->addShuffle(*inputEmbeddedData);
        ASSERT(shuffleLayer != nullptr);
        shuffleLayer->setName("Reshape input for LSTM decoder");
        nvinfer1::Dims shuffleDims{3, {beamWidth, 1, inputWidth}};
        shuffleLayer->setReshapeDimensions(shuffleDims);
        shuffledInput = shuffleLayer->getOutput(0);
        ASSERT(shuffledInput != nullptr);
    }

    auto decoderLayer = network->addRNNv2(*shuffledInput, mNumLayers, mNumUnits, 1, nvinfer1::RNNOperation::kLSTM);
    ASSERT(decoderLayer != nullptr);
    decoderLayer->setName("LSTM decoder");

    decoderLayer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    decoderLayer->setDirection(nvinfer1::RNNDirection::kUNIDIRECTION);

    std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kFORGET, nvinfer1::RNNGateType::kINPUT,
        nvinfer1::RNNGateType::kCELL, nvinfer1::RNNGateType::kOUTPUT});
    for (size_t i = 0; i < mGateKernelWeights.size(); i++)
    {
        // we have 4 + 4 gates
        bool isW = ((i % 8) < 4);
        decoderLayer->setWeightsForGate(i / 8, gateOrder[i % 4], isW, mGateKernelWeights[i]);
        decoderLayer->setBiasForGate(i / 8, gateOrder[i % 4], isW, mGateBiasWeights[i]);
    }

    decoderLayer->setHiddenState(*inputStates[0]);
    decoderLayer->setCellState(*inputStates[1]);
    *outputData = decoderLayer->getOutput(0);
    ASSERT(*outputData != nullptr);

    {
        auto shuffleLayer = network->addShuffle(**outputData);
        ASSERT(shuffleLayer != nullptr);
        shuffleLayer->setName("Reshape output from LSTM decoder");
        nvinfer1::Dims shuffleDims{2, {beamWidth, mNumUnits}};
        shuffleLayer->setReshapeDimensions(shuffleDims);
        auto shuffledOutput = shuffleLayer->getOutput(0);
        ASSERT(shuffledOutput != nullptr);
        *outputData = shuffledOutput;
    }

    // Per layer hidden output
    outputStates[0] = decoderLayer->getOutput(1);
    ASSERT(outputStates[0] != nullptr);

    // Per layer cell output
    outputStates[1] = decoderLayer->getOutput(2);
    ASSERT(outputStates[1] != nullptr);
}

std::vector<nvinfer1::Dims> LSTMDecoder::getStateSizes()
{
    nvinfer1::Dims hiddenStateDims{2, {mNumLayers, mNumUnits}};
    nvinfer1::Dims cellStateDims{2, {mNumLayers, mNumUnits}};
    return std::vector<nvinfer1::Dims>({hiddenStateDims, cellStateDims});
}

std::string LSTMDecoder::getInfo()
{
    std::stringstream ss;
    ss << "LSTM Decoder, num layers = " << mNumLayers << ", num units = " << mNumUnits;
    return ss.str();
}
} // namespace nmtSample
