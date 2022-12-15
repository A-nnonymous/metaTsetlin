// An implementation of Tsetlin Machine in C++ using SIMD instructions and meta-heuristic optimizers.

// The MIT License (MIT)
// Copyright (c) 2022 Pan Zhaowu <panzhaowu21s@ict.ac.cn>

//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#include "io.h"
using std::vector;
using std::string;

/// @brief Convert continuous data to discrete through grey-scale like threshold.
/// @param threshold Vector of incremental continuous threshold.
/// @param raw Raw continuous data.
/// @return A vector of discrete data.
vector<int> getDiscreteResponse(vector<double> threshold, double raw)
{
    vector<int> result(threshold.size()+1, 0);
    for (int i = 0; i < threshold.size(); i++)
    {
        double thiscut = threshold[i];
        if(raw <= thiscut)
        {
            result[i] = 1;
            return result;
        }
    }
    result[result.size()-1] = 1; // larger than final cut.
    return result;
}
vector<string> threshold2Tags(vector<double> thresholds, bool isAscent)
{
    auto spanNum = thresholds.size() + 1;
    vector<string> tags(spanNum);
    tags[0] = (isAscent? "<":">") + std::to_string(thresholds[0]);
    string prevThreshold = std::to_string(thresholds[0]);

    for (int i = 1; i < spanNum - 1; i++)
    {
        string thisThreshold = std::to_string(thresholds[i]);
        string thisTag = prevThreshold + " ~ " + thisThreshold;
        tags[i] = thisTag;
        prevThreshold = thisThreshold;
    }
    tags[spanNum - 1] = (isAscent? ">":"<") + std::to_string(thresholds[spanNum - 2]);
    return tags;
}

/// @brief Save Tsetlin machine model in binary format.
/// @param machine Target model.
/// @param outputPath Path of output file.
void saveModel( TsetlinMachine::model   &machine,
                string                  outputPath)
{
    write_binary<TsetlinMachine::model>(&machine,1,outputPath);
}

/// @brief Load Tsetlin machine model from binary model file.
/// @param modelPath Path of model file.
/// @return A structured model of Tsetlin machine.
TsetlinMachine::model loadModel(string modelPath)
{
    TsetlinMachine::model result;
    read_binary<TsetlinMachine::model>(modelPath, &result);
    return result;
}

