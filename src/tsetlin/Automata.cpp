#include "Automata.h"


Automata::Automata(AutomataArgs args, vector<vector<int>> &input, vector<int> &target):
_no(args.no),
_clauseNum(args.clauseNum),
_T(args.T),
_sLow(args.sLow),
_sHigh(args.sHigh),
_dropoutRatio(args.dropoutRatio),
_sharedInputData(input),
_targets(target) // Target vector is arranged to 2D vector, each row is an reflection of a sigle dimension of output vector.
{
    _rng = std::mt19937(std::random_device{}());
    _voteSum = 0;
    Clause::ClauseArgs cArgs;
    cArgs.inputSize = args.inputSize;
    for(int i = 0; i< args.clauseNum; i++)
    {
        cArgs.no = i;
        cArgs.specificity = _sLow + i * (_sHigh - _sLow)/((double)_clauseNum);
        
        cArgs.isPositiveClause = true;
        Clause posClause(cArgs);
        _positiveClauses.push_back(std::move(posClause));

        cArgs.isPositiveClause = false;
        Clause negClause(cArgs);
        _negativeClauses.push_back(std::move(negClause));
    }
}


/// @brief Check the integrety of target model.
/// @param targetModel Input model.
/// @return Result of this check.
bool Automata::modelIntegrityCheck(model targetModel)
{
    bool isRightLength =    (targetModel.negativeClauses.size() == _clauseNum) &&
                            (targetModel.positiveClauses.size() == _clauseNum);
    bool isRightPlace = (targetModel.no == _no);
    return isRightLength && isRightPlace;
}

/// @brief Check and import Automata model
/// @param targetModel Model depacked and passed from its caller.
void Automata::importModel(model targetModel)
{
    if(!modelIntegrityCheck(targetModel))
    {
        std::cout<< "Automata model integrity check failed at automata"<<_no<<std::endl;
        throw;return;
    }
    for (int i = 0; i < _clauseNum; i++)
    {
        _positiveClauses[i].importModel(targetModel.positiveClauses[i]);
        _negativeClauses[i].importModel(targetModel.negativeClauses[i]);
    }
}

/// @brief Calling every clause to export their automatons' state and pack them to the caller.
/// @return Packed struct of Automata model.
Automata::model Automata::exportModel()
{
    Automata::model result;
    result.no = _no;
    result.positiveClauses.resize(_clauseNum,Clause::model());
    result.negativeClauses.resize(_clauseNum,Clause::model());
    for (int i = 0; i < _clauseNum; i++)
    {
        result.positiveClauses[i] = _positiveClauses[i].exportModel();
        result.negativeClauses[i] = _negativeClauses[i].exportModel();
    }
    return result;
}


/// @brief Forward function, doing vote for learning or predicting.
/// @param datavec A single vector of input data containing _inputSize number of elements.
/// @return Result of all clauses' vote.
int Automata::forward(vector<int> &datavec)
{
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
void Automata::backward(int response)
{
    int     clampedSum = std::min(_T, std::max(-_T, response));
    double   rescaleFactor = 1.0f / static_cast<double>(2 * _T);

    double probFeedBack0 = (_T - clampedSum) * rescaleFactor; // The larger the T is, the less biased sum influences
    double probFeedBack1 = (_T + clampedSum) * rescaleFactor;
    
    static std::discrete_distribution<> probChoice({probFeedBack0, probFeedBack1});
    static std::discrete_distribution<> dropout({_dropoutRatio, 1-_dropoutRatio});
    static vector<bool> actP0(_clauseNum,false);
    static vector<bool> actP1(_clauseNum,false);
    static vector<bool> pick(_clauseNum,false);
    
    for (int i = 0; i < _clauseNum; i++)
    {
        actP0[i] = (probChoice(_rng) == 0);          // Generate action vector with possibility of probFeedBack0.
        actP1[i] = (probChoice(_rng) == 1);
        pick[i] = (dropout(_rng) == 1);
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


/// @brief Learning process including forward and backward of a single epoch.
void Automata::learn()
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
Automata::predict (vector<vector<int>> &input)
{
    vector<Prediction> result(input.size(),Prediction());
    for (int i = 0; i < input.size(); i++)
    {
        Prediction thisPrediction;
        int sum = forward(input[i]);
        thisPrediction.result = (sum>0? 1:0);
        thisPrediction.confidence = sum/(double)_clauseNum;
        result[i] = thisPrediction;
    }
    return result;
}