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

#ifndef SAMPLE_NMT_PROJECTION_
#define SAMPLE_NMT_PROJECTION_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Projection
 *
 * \brief calculates raw logits
 *
 */
class Projection : public Component
{
public:
    typedef std::shared_ptr<Projection> ptr;

    Projection() = default;

    /**
     * \brief add raw logits to the network
     */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input, nvinfer1::ITensor** outputLogits)
        = 0;

    /**
     * \brief get the size of raw logits vector
     */
    virtual int32_t getOutputSize() = 0;

    ~Projection() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_PROJECTION_
