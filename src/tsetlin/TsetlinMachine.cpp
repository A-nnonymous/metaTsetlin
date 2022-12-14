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

#include "TsetlinMachine.h"
#include <thread>

TsetlinMachine::TsetlinMachine( MachineArgs args, vector<string> tierTags)noexcept:
_inputSize(args.inputSize),
_outputSize(args.outputSize),
_clausePerOutput(args.clausePerOutput),
_T(args.T),
_sLow(args.sLow), _sHigh(args.sHigh),
_dropoutRatio(args.dropoutRatio),
_myArgs(args),
_tierTags(tierTags)
{
    Automata::AutomataArgs aArgs;
    aArgs.clauseNum = _clausePerOutput;
    aArgs.dropoutRatio = _dropoutRatio;
    aArgs.inputSize = _inputSize;
    aArgs.sLow = _sLow;
    aArgs.sHigh = _sHigh;
    aArgs.T = _T;

    _response.resize(_outputSize, vector<int>(1,0));    // Set dummy zero response as placeholder.
    for (int i = 0; i < _outputSize; i++)
    {
        aArgs.no = i;
        Automata thisAutomata(aArgs,_sharedData,_response[i]);
        _automatas.push_back(thisAutomata);
    }
}


/// @brief Check model integrity before importing
/// @param targetModel Model that user intend to import
/// @return Boolean value of the integrity
bool
TsetlinMachine::modelIntegrityCheck(model &targetModel)
{
    bool isRightArgument =  (targetModel.modelArgs == _myArgs);
    bool isRightAutomatas = (targetModel.automatas.size() == _outputSize);
    
    return isRightArgument && isRightAutomatas;
}

/// @brief Check the integrity of argument 'data'.
/// @param response Input unknown size 2D vector.
/// @return Result of integrity check procedure.
bool
TsetlinMachine::dataIntegrityCheck(const vector<vector<int>> &data)
{
    bool isZeroSize = (data.size()==0);
    bool isCorrectLength = true;
    for (int i = 0; i < data.size(); i++)
    {
        isCorrectLength &= (data[i].size() == _inputSize);
        if(!isCorrectLength)break;
    }
    bool result = (!isZeroSize) && (isCorrectLength);
    if (!result)
    {
        std::cout<<"Data failed integrity check."<<std::endl;
    }
    
    return result;
}

/// @brief Check the integrity of argument 'response'.
/// @param response Input unknown size 2D vector.
/// @return Result of integrity check procedure.
bool
TsetlinMachine::responseIntegrityCheck(const vector<vector<int>> &response)
{
    bool isZeroSize = (response.size()==0);
    bool isCorrectLength = (response.size()==_outputSize);

    bool result = (!isZeroSize) && (isCorrectLength); 
    if (!result)
    {
        std::cout<<"Response failed integrity check."<<std::endl;
    }
    return result;
}

/// @brief Transpose response matrix before store in shared vector.
/// @param original Original 2D vector of response, shaped in ( sampleNum * _outputSize )
/// @return Transposed 2D vector of response , shaped in ( _outputSize * sampleNum )
vector<vector<int>> 
TsetlinMachine::transpose(vector<vector<int>> &original)
{
    int rowNum = original.size();
    int colNum = original[0].size();
    vector<vector<int>> result(colNum, vector<int>(rowNum,0));
    for (int row = 0; row < rowNum; row++)
    {
        for (int col = 0; col < colNum; col++)
        {
            result[col][row] = original[row][col];
        }
    }
    return result;    
}

/// @brief Pack and 'align' the original vector of int to 512Byte pack with zero-padding if size not equal to 16-mer.
/// @param original Original vector of 32bit integer.
/// @return Vector of packed and zero-padded __m512i pack vector.
__attribute__((target("avx512f")))
vector<__m512i>
TsetlinMachine::pack(vector<int> &original)
{
    int packNum = original.size()/16 + (original.size()%16==0? 0:1);
    alignas(64) struct pack{
        int data[16];
        pack(){
            for (int i = 0; i < 16; i++)
            {
                data[i] = 0;
            }
            
        }
    };
    vector<__m512i> result(packNum, _mm512_set1_epi32(0));
    for (int i = 0; i < packNum; i++)
    {
        pack thisPack;
        for (int j = 0; j < 16; j++)
        {
            thisPack.data[j] = (i*16+j)<(original.size())? original[i*16+j] : 0;
        }
        
        result[i] = _mm512_loadu_epi32(&thisPack);
    }
    return result;
}

/*
/// @brief Import model from user.
/// @param targetModel Target model in class of TsetlinMachine::model.
void
TsetlinMachine::importModel(model &targetModel)
{
    if(!TsetlinMachine::modelIntegrityCheck(targetModel))
    {
        std::cout<<"Your Tsetlin Machine model failed integrity check!"<<std::endl;
        throw; return;
    }
    for (int i = 0; i < _outputSize; i++)
    {
        _automatas[i].importModel(targetModel.automatas[i]);
    }
}
*/


/// @brief Export current model.
/// @return Current model and arguments.
TsetlinMachine::model
TsetlinMachine::exportModel()
{
    TsetlinMachine::model result;
    result.modelArgs = _myArgs;
    result.tierTags = _tierTags;
    result.automatas.resize(_outputSize, Automata::model());
    for (int i = 0; i < _outputSize; i++)
    {
        result.automatas[i] = _automatas[i].exportModel();
    }
    return result;
}

/// @brief Perform data integrity check and load into shared vector.
/// @param data 2D vector shaped in ( sampleNum * _inputSize )
/// @param response 2D vector shaped in ( sampleNum * _outputSize )
void
TsetlinMachine::load(vector<vector<int>> &data,
                                vector<vector<int>> &response)
{
    vector<vector<int>> temp = transpose(response);
    if( !dataIntegrityCheck(data) || 
        !responseIntegrityCheck(temp)) {throw;return;}
    _sharedData.resize( data.size(),
                        vector<__m512i>(1, _mm512_set1_epi32(0)));
    for (int i = 0; i < _outputSize; i++)
    {
        _response[i] = temp[i];
    }

    for (int i = 0; i < data.size(); i++)
    {
        _sharedData[i] = pack(data[i]);
    }
    _response.shrink_to_fit();
    _sharedData.shrink_to_fit();
    //std::cout<<"Loaded "<<data.size()<< " samples, each consumes "<<_sharedData[0].size()<< " blocks"<<std::endl;
}

/// @brief Train this Tsetlin machine using loaded data.
/// @param epoch Max count of repeat training time.
void
TsetlinMachine::train(int epoch)
{
    for (int i = 0; i < epoch; i++)
    {
        for (int j = 0; j < _outputSize; j++)   // Each output corresponds an automata.
        {
            _automatas[j].learn();
        }
    }
}

/// @brief Load data and predict response using trained tsetlin machine.
/// @param data 2D vector shaped in ( sampleNum * _inputSize )
/// @return 2D vector shaped in ( sampleNum * _outputSize )
vector<vector<int>>
TsetlinMachine::loadAndPredict(vector<vector<int>> &data)
{
    if( !dataIntegrityCheck(data)) throw;
    vector<vector<Automata::Prediction>> prediction(_outputSize,
                                                    vector<Automata::Prediction>(data.size(),Automata::Prediction()));
    vector<vector<__m512i>> mdata;
    mdata.resize( data.size(),
                        vector<__m512i>(1, _mm512_set1_epi32(0)));
    for (int i = 0; i < data.size(); i++)
    {
        mdata[i] = pack(data[i]);
    }

    for (int i = 0; i < _outputSize; i++)
    {
        prediction[i] = _automatas[i].predict(mdata);
    }
    vector<vector<int>> result(data.size(), vector<int>(_outputSize,0));
    for (int sampleIdx = 0; sampleIdx < data.size(); sampleIdx++)
    {
        double  maxConfidence = 0;
        int     competitorIdx = 0;
        for (int featureIdx = 0; featureIdx < _outputSize; featureIdx++)
        {
            if(prediction[featureIdx][sampleIdx].confidence > maxConfidence)
            {
                competitorIdx = featureIdx;
                maxConfidence = prediction[featureIdx][sampleIdx].confidence;
            }
        }
        result[sampleIdx][competitorIdx] = 1;
    }
    return result;
}