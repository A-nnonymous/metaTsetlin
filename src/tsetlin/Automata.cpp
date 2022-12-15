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

#include "Automata.h"
#include <thread>


Automata::Automata(AutomataArgs args, vector<vector<__m512i>> &input, vector<int> &target)noexcept:
_no(args.no),
_inputSize(args.inputSize),
_clauseNum(args.clauseNum),
_T(args.T),
_sLow(args.sLow),
_sHigh(args.sHigh),
_dropoutRatio(args.dropoutRatio),
_sharedInputData(input),
_targets(target)
{
    static pcg_extras::seed_seq_from<std::random_device> seed_source;
    static pcg64_fast _rng(seed_source);
    _voteSum = 0;
    Clause::ClauseArgs cArgs;

    cArgs.inputSize = args.inputSize;
    for(int i = 0; i< args.clauseNum; i++)
    {
        cArgs.no = i;
        cArgs.specificity = _sLow + i * (_sHigh - _sLow)/((double)_clauseNum);
        
        Clause temp(cArgs);
        _positiveClauses.push_back(temp);
        _negativeClauses.push_back(std::move(temp));
    }
}


/// @brief Forward function, doing vote for learning or predicting.
/// @param datavec A single vector of input data containing _inputSize number of elements.
/// @return Result of all clauses' vote.
int Automata::forward(vector<__m512i> &datavec)noexcept
{
    //std::cout<< "Start forwarding, each sample consume "<< datavec.size()<<std::endl;
    int sum = 0;
    for(auto    pos = _positiveClauses.begin();
                pos < _positiveClauses.end();
                pos ++)
    {
        sum+=pos->vote(datavec);
    }
    for (auto   neg = _negativeClauses.begin();
                neg < _negativeClauses.end();
                neg++)
    {
        sum-=neg->vote(datavec);
    }
    return sum;
}


/// @brief Backward function, containing arrangement of two types of feedback.
/// @param response Target response of this input vector.
void Automata::backward(int &response)noexcept
{
    int     clampedSum = std::min(_T, std::max(-_T, response));
    double   rescaleFactor = 1.0f / static_cast<double>(2 * _T);

    double probFeedBack0 = (_T - clampedSum) * rescaleFactor; // The larger the T is, the less biased sum influences
    double probFeedBack1 = (_T + clampedSum) * rescaleFactor;
    std::discrete_distribution<> probChoice({probFeedBack0, probFeedBack1});
    std::discrete_distribution<> dropout({_dropoutRatio, 1-_dropoutRatio});
    vector<bool> actP0(_clauseNum,false);
    vector<bool> actP1(_clauseNum,false);
    vector<bool> pick(_clauseNum,false);
    
    for (int i = 0; i < _clauseNum; i++)
    {
        actP0[i] = !probChoice(_rng);          // Generate action vector with possibility of probFeedBack0.
        actP1[i] = probChoice(_rng);
        pick[i] = dropout(_rng);
    }

    for (int i = 0; i < _clauseNum; i++)
    {
        if((response==1) && actP0[i] && pick[i])
        {
            _positiveClauses[i].feedbackTypeI();
            _negativeClauses[i].feedbackTypeII();
        }
        if((response==0) && actP1[i] && pick[i])
        {
            _positiveClauses[i].feedbackTypeII();
            _negativeClauses[i].feedbackTypeI();
        }
    }
}

/// @brief Check the integrety of target model.
/// @param targetModel Input model.
/// @return Result of this check.
bool Automata::modelIntegrityCheck(model &targetModel)
{
    bool isRightLength =    (targetModel.negativeClauses.size() == _clauseNum) &&
                            (targetModel.positiveClauses.size() == _clauseNum);
    return isRightLength;
}

/// @brief Learning process including forward and backward of a single epoch.
void Automata::learn()noexcept
{
    for (int i = 0; i < _sharedInputData.size(); i++)
    {
        forward(_sharedInputData[i]);
        backward(_targets[i]);
    }
}

/// @brief Generate output using learned clauses in this automata
/// @param input Given input 2D vector, shaped in ( sampleNum * _inputSize )
/// @return Vector of prediction structs, containing result of each example and it's predict confidence.
vector<Automata::Prediction>
Automata::predict (vector<vector<__m512i>> &input)noexcept
{
    vector<Prediction> result(input.size(),Prediction());
    for (int i = 0; i < input.size(); i++)
    {
        Prediction thisPrediction;
        int sum = forward(input[i]);
        thisPrediction.result = (sum>0? 1:0);
        thisPrediction.confidence = sum/(double)_clauseNum;
        //std::cout<< "Automata "<<_no<<" prediction "<< i <<"is "<< thisPrediction.result<<" with confidence of: "<< thisPrediction.confidence<<std::endl;
        result[i] = thisPrediction;
    }
    return result;
}

Automata::model Automata::exportModel()
{
    model result;
    vector<vector<int>> pos, neg;
    pos.resize(_clauseNum);
    neg.resize(_clauseNum);
    for (int i = 0; i < _clauseNum; i++)
    {
        pos[i] = _positiveClauses[i].exportModel();
        neg[i] = _negativeClauses[i].exportModel();
    }
    result.positiveClauses = pos;
    result.negativeClauses = neg;
    return result;
}

/*
void Automata::importModel(model &targetModel)
{
    if(!Automata::modelIntegrityCheck(targetModel))
    {
        std::cout<<"Your Tsetlin Machine model failed integrity check!"<<std::endl;
        throw; return;
    }
    for (int i = 0; i < _clauseNum; i++)
    {
        _positiveClauses[i].importModel(targetModel.positiveClauses[i]);
        _negativeClauses[i].importModel(targetModel.negativeClauses[i]);
    }
}
*/