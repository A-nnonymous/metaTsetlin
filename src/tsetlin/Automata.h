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

#include "Clause.h"
using std::vector;

/// @brief A tsetlin automata is fundamental object to learn a digit of output from input.
class Automata{
public:
    struct AutomataArgs
    {
        int     no;
        int     inputSize;
        int     clauseNum;
        int     T;
        double  sLow, sHigh;
        double  dropoutRatio;
    };
    struct Prediction
    {
        int     result;
        double  confidence;
        Prediction()
        {
            result = 0;
            confidence = 0;
        }
    };
    struct model
    {
        vector<vector<int>>   positiveClauses;// Arranged in size of ClauseNum * (literalNum * 2)
        vector<vector<int>>   negativeClauses;
        model(){}
    };
    

private:
    const int                   _no;
    const int                   _inputSize;
    const int                   _clauseNum;         // Attention, this is only represent one single polarity.
    const int                   _T;
    const double                _sLow;
    const double                _sHigh;             // This is for multigranular clauses.
    const double                _dropoutRatio;      // Random dropout some clauses.
    vector<vector<__m512i>>     &_sharedInputData;  // When start traning, reference dataset from TM.
    vector<int>                 &_targets;

    pcg64_fast                  _rng;
    int                         _voteSum;           // Sum of all clauses' vote.
    vector<Clause>              _positiveClauses;
    vector<Clause>              _negativeClauses;

    int     forward(vector<__m512i> &datavec)noexcept;
    void    backward(int &response)noexcept;
    bool    modelIntegrityCheck(model &targetModel);
public:
    Automata(AutomataArgs args, vector<vector<__m512i>> &input, vector<int> &target)noexcept;

    void                learn()noexcept;
    vector<Prediction>  predict(vector<vector<__m512i>> &input)noexcept;

    model               exportModel();
    //void                importModel(model &targetModel);
};