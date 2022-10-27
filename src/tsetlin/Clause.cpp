#include "Clause.h"
using std::vector;


Clause::Clause(ClauseArgs args):
_no(args.no),
_isPositiveClause(args.isPositiveClause),
_literalNum(args.inputSize),
_s(args.specificity), _sInv(1.0/_s), _sInvConj(1.0-_sInv)
{
    _rng = std::mt19937(std::random_device{}());
    _vote = 0;  _isVoteDirty = false;
    _positiveLiterals.resize(_literalNum,0);
    _posInclusionMask.resize(_literalNum,true);
    _negativeLiterals.resize(_literalNum,0);
    _negInclusionMask.resize(_literalNum,true);
    _inputMask.resize(_literalNum,false);
}

bool Clause::modelIntegrityCheck(model targetModel)
{
    bool isRightLength =    (targetModel.positiveLiteral.size() == _literalNum) &&
                            (targetModel.negativeLiteral.size() == _literalNum);
    bool isRightPlace = (targetModel.no == _no);
    return isRightLength && isRightPlace;
}

void Clause::importModel(model targetModel)
{
    if(!modelIntegrityCheck(targetModel))
    {
        std::cout<< "Clause model failed integrity check at clause "<<_no<<std::endl;
        throw; return;
    }
    _positiveLiterals = targetModel.positiveLiteral;
    _negativeLiterals = targetModel.negativeLiteral;
}

Clause::model Clause::exportModel()
{
    Clause::model result;
    result.no = _no;
    result.positiveLiteral = _positiveLiterals;
    result.negativeLiteral = _negativeLiterals;
    return result;
}

/// @brief Vote function used for both train and predict procedure.
/// @param in Data vector that in the shape of ( 1, _literalNum )
/// @return Vote result, 0 or 1.
int Clause::vote(vector<int> in)
{
    if(in.size() != _literalNum)        // Can't happen except some panic programming by myself.
    {
        std::cout<<"Input vector size not fit for Clause."<<std::endl;
        throw;
    }
    
    bool posResult = true, negResult = true;
    for (int i = 0; i < _literalNum; i++)           // Run once per data input.
    {
        _posInclusionMask[i] = (_positiveLiterals[i] > 0);
        _negInclusionMask[i] = (_negativeLiterals[i] > 0);
        _inputMask[i] = (in[i] > 0);
    }
    for (int i = 0; i < _literalNum; i++)
    {
        posResult &= _posInclusionMask[i] && _inputMask[i] || !_posInclusionMask[i];
        negResult &= _negInclusionMask[i] && (!_inputMask[i]) || !_negInclusionMask[i];
        if(!posResult || !negResult) break;         // Break when first unsatisfied literal occured.
    }
    int result = (posResult&&negResult? 1:0);
    _vote = result; _isVoteDirty = true;
    return result;
}

/// @brief Reinforce positive and negative literals according to 's' ,input, previous vote.
/// @param in Previous input vector.
void Clause::feedbackTypeI()
{
    if(!_isVoteDirty)       // Vote is untouched before feedback, must not happen.
    {
        std::cout<<"Panicking, haven't vote before this feedback action!"<<std::endl;
        throw;
    }
    static std::discrete_distribution<> d({_sInv, _sInvConj});
    static vector<bool> radical(_literalNum,0);
    static vector<bool> conservative(_literalNum,0);
    for (int i = 0; i < _literalNum; i++)
    {
        radical[i] =        (d(_rng)==1);   // Fill 'true' with possibility of _sInvConj
        conservative[i] =   (d(_rng)==0);   // Complement possibility, but not correlated to radical[].
    }
    
    if(_vote)
    {
        for(int i=0; i<_literalNum; i++)
        {
            if(_inputMask[i])
            {
                _positiveLiterals[i] += radical[i];         // Good input for positive literal, reinforce inclusion radically.
                _negativeLiterals[i] -= conservative[i];    // Bad input for negative literal, reinforce exclusion conservatively.
            }
            else
            {
                _negativeLiterals[i] += radical[i];         // Vice versa.
                _positiveLiterals[i] -= conservative[i];
            }
        }
    }
    else
    {
        for(int i=0; i<_literalNum; i++)
        {
            _positiveLiterals[i] -= conservative[i];
            _negativeLiterals[i] -= !(radical[i]);      // Use reverse of radical to prevent correspondance.
        }
    }
    _isVoteDirty = false;
}

/// @brief Reinforce positive and negative literals according to inclusion, input, previous vote.
/// @param in Previous input vector.
void Clause::feedbackTypeII()
{
    if(!_isVoteDirty)       // Vote is untouched before feedback, must not happen.
    {
        std::cout<<"Panicking, haven't vote before this feedback action!"<<std::endl;
        throw;
    }
    if(_vote==0)return;
    for (int i = 0; i < _literalNum; i++)
    {
        _positiveLiterals[i] += (!_posInclusionMask[i])&&(!_inputMask[i]);
        _negativeLiterals[i] += (!_negInclusionMask[i])&&(_inputMask[i]);
    }
    _isVoteDirty = false;
}
