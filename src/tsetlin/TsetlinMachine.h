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

#pragma once
#include "Automata.h"
using std::vector;
using std::string;

// TODO: Add boost::serialization to export full model
class TsetlinMachine{
public:
    struct MachineArgs
    {
        int             inputSize;
        int             outputSize;
        int             clausePerOutput;
        int             T;
        double          sLow, sHigh;
        double          dropoutRatio;
        vector<string>  tierTags;

        bool operator==(MachineArgs a)const
        {
            return  (a.clausePerOutput = this->clausePerOutput) &&
                    (a.dropoutRatio = this->dropoutRatio) &&
                    (a.inputSize = this->inputSize) &&
                    (a.outputSize = this->outputSize)&&
                    (a.sHigh = this->sHigh)&&
                    (a.sLow = this->sLow) &&
                    (a.T = this->T);
        }
    };
    struct model
    {
        MachineArgs             modelArgs;
        vector<string>          tierTags;
        vector<Automata::model> automatas;
        model(){}
    };
    
private:
    const int                   _inputSize;
    const int                   _outputSize;
    const int                   _clausePerOutput;
    const int                   _T;
    const double                _sLow, _sHigh;
    const double                _dropoutRatio;
    const MachineArgs           _myArgs;
    const vector<string>        _tierTags;

    vector<Automata>            _automatas;
    
    vector<vector<__m512i>>     _sharedData;
    
    vector<vector<int>>         _response;      // Each row is a reflection of multi-dimensional dataset.

    bool    modelIntegrityCheck(model &targetModel);
    bool    dataIntegrityCheck( const vector<vector<int>> &data);
    bool    responseIntegrityCheck(const vector<vector<int>> &response);

    vector<vector<int>> transpose(vector<vector<int>> &original);
    
    
    vector<__m512i>     pack(vector<int> &original);

public:
    TsetlinMachine( MachineArgs args, vector<string> tierTags)noexcept;
    //TsetlinMachine( model &savedModel)noexcept;

    void                load(   vector<vector<int>> &data,
                                vector<vector<int>> &response);
    void                train(int epoch);
    
    vector<vector<int>> loadAndPredict(vector<vector<int>> &data);

    //void                importModel(model &targetModel);
    model               exportModel();
};