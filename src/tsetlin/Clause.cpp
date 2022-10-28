#include "Clause.h"
using std::vector;


Clause::Clause(ClauseArgs args):
_no(args.no),
_isPositiveClause(args.isPositiveClause),
_literalNum(args.inputSize),
_s(args.specificity), _sInv(1.0/_s), _sInvConj(1.0-_sInv),
_blockNum(args.inputSize/16 + (args.inputSize%16==0? 0:1))
{
    __m512i     ones = _mm512_set1_epi32(1);
    __m512i     zeros= _mm512_set1_epi32(0);
    __mmask16   zeroMask = _mm512_cmpeq_epi32_mask(ones,zeros);
    __mmask16   oneMask = _mm512_cmpeq_epi32_mask(ones,ones);

    _rng = std::mt19937(std::random_device{}());
    _vote = 0;  _isVoteDirty = false;
    _positiveLiteralBlocks.resize(_blockNum, zeros);
    _negativeLiteralBlocks.resize(_blockNum, zeros);
    _posInclusionMaskBlocks.resize(_blockNum, oneMask);
    _negInclusionMaskBlocks.resize(_blockNum, oneMask);
    _inputMaskBlocks.resize(_blockNum, zeroMask);
    _inputMaskBlocksInverse.resize(_blockNum, zeroMask);
}

/// @brief Check model integrity before importing
/// @param targetModel Model that user intend to import
/// @return Boolean value of the integrity
bool Clause::modelIntegrityCheck(model targetModel)
{
    bool isRightLength =    (targetModel.positiveLiteral.size() == _literalNum) &&
                            (targetModel.negativeLiteral.size() == _literalNum);
    bool isRightPlace = (targetModel.no == _no);
    return isRightLength && isRightPlace;
}

/// @brief Pack and 'align' the original vector of int to 512Byte pack with zero-padding if size not equal to 16-mer.
/// @param original Original vector of 32bit integer.
/// @return Vector of packed and zero-padded __m512i pack vector.
vector<__m512i>
Clause::pack(vector<int> original)
{
    int packNum = original.size()/16 + (original.size()%16==0? 0:1);
    vector<__m512i> result(packNum, _mm512_set1_epi32(0));
    for (int i = 0; i < packNum; i++)
    {
        result[i] = _mm512_loadu_epi32(&original[i*16]);
    }
    return result;
}

/// @brief Unpack the vector of 512Byte integer packs
/// @param original Original vector of 512Byte integer packs,each element contain 16 int32 number.
/// @return A vector of unpacked int that may contains padded elements.
vector<int>
Clause::unpack(vector<__m512i> original)
{
    int length = original.size() *16;
    vector<int> result(length,0);
    for (int i = 0; i < original.size(); i++)
    {
        _mm512_storeu_epi32(&result[i*16],original[i]);
    }
    return result;
}
/// @brief Vote function used for both train and predict procedure.
/// @param in Data vector that in the shape of ( 1, _literalNum )
/// @return Vote result, 0 or 1.
int Clause::vote(vector<__m512i> in)
{
    if(in.size() != _blockNum)        // Can't happen except some panic programming by myself.
    {
        //std::cout<<"Input vector size is: "<< in.size()<<" not fit for Clause."<<std::endl;
        throw;
    }
    
    static __m512i      zeros = _mm512_set1_epi32(0);
    static __m512i      ones = _mm512_set1_epi32(1);
    static __mmask16    zeroMask = _mm512_cmpeq_epi32_mask(ones,zeros);
    vector<__mmask16>   posWrong(_blockNum, zeroMask);
    vector<__mmask16>   negWrong(_blockNum, zeroMask);

    for (int i = 0; i < _blockNum; i++)           // Run once per data input.
    {
        _posInclusionMaskBlocks[i] = _mm512_cmpge_epi32_mask(_positiveLiteralBlocks[i], zeros);
        _negInclusionMaskBlocks[i] = _mm512_cmpge_epi32_mask(_positiveLiteralBlocks[i], zeros);
        _inputMaskBlocks[i] = _mm512_cmpgt_epi32_mask(in[i], zeros);                // Greater than zero (only possible value is one)
        _inputMaskBlocksInverse[i] = _knot_mask16(_inputMaskBlocks[i]);
    }
    bool posNoProblem = true, negNoProblem= true;
    bool hasProblem;
    for (int i = 0; i < _blockNum; i++)
    {
        posWrong[i] =  _kor_mask16(posWrong[i],
                                                _kand_mask16(_posInclusionMaskBlocks[i],_inputMaskBlocksInverse[i]));    // Included but input=0, positive literal wrong.
        //posResult &= _posInclusionMask[i] && _inputMask[i] || !_posInclusionMask[i];
        negWrong[i] =  _kor_mask16(negWrong[i],
                                                _kand_mask16(_negInclusionMaskBlocks[i],_inputMaskBlocks[i]));                  // Included but input=1, negative literal wrong.
        //negResult &= _negInclusionMask[i] && (!_inputMask[i]) || !_negInclusionMask[i];
        posNoProblem = (_mm512_mask2int(posWrong[i]) == 0 );    // So far so good.
        negNoProblem = (_mm512_mask2int(negWrong[i]) == 0 );
        hasProblem = !(posNoProblem && negNoProblem);
        if(hasProblem) break;         // Break when first unsatisfied literal occured.
    }
    // Warning: boundary problem caused by padding input must be considered.
    int result = (hasProblem? 0:1);
    _vote = result; _isVoteDirty = true;
    return result;
}

/// @brief Reinforce positive and negative literals according to 's' ,input, previous vote.
void Clause::feedbackTypeI()
{
    if(!_isVoteDirty)       // Vote is untouched before feedback, must not happen.
    {
        std::cout<<"Panicking, haven't vote before this feedback action!"<<std::endl;
        throw;
    }
    static std::discrete_distribution<> d({_sInv, _sInvConj});
    static vector<int>                  radical(_literalNum,0);
    static vector<int>                  conservative(_literalNum,0);
    static __m512i                      zeros = _mm512_set1_epi32(0);
    static __m512i                      ones = _mm512_set1_epi32(1);
    static __m512i                      negOnes= _mm512_set1_epi32(-1);
    static __mmask16                    zeroMask = _mm512_cmpeq_epi32_mask(ones,zeros);
    static vector<__m512i>              radicalBlock(_blockNum, zeros);
    static vector<__m512i>              conservativeBlock(_blockNum, zeros);
    static vector<__mmask16>            radicalPosMaskBlock(_blockNum, zeroMask);
    static vector<__mmask16>            conservativeNegMaskBlock(_blockNum, zeroMask);

    for (int i = 0; i < _literalNum; i++)
    {
        radical[i] =        (d(_rng)==1);   // Fill 'true' with possibility of _sInvConj
        conservative[i] =   (d(_rng)==0);   // Complement possibility, but not correlated to radical[].
    }
    radicalBlock = pack(radical); conservativeBlock = pack(conservative);

    for (int i = 0; i < _blockNum; i++)
    {
        radicalPosMaskBlock[i] = _mm512_cmpeq_epi32_mask(radicalBlock[i], ones);
        conservativeNegMaskBlock[i] = _mm512_cmpeq_epi32_mask(conservativeBlock[i], ones);
    }
    
    if(_vote)
    {
        /*
        for(int i=0; i<_literalNum; i++)
        {
            if(_inputMask[i])
            {
                _positiveLiterals[i] += radical[i];         // Good input for positive literal, reinforce inclusion radically.
                _negativeLiterals[i] -= conservative[i];    // Bad input for negative literal, reinforce exclusion conservatively.
            }
            else
            {
                _positiveLiterals[i] -= conservative[i];
                _negativeLiterals[i] += radical[i];         // Vice versa.
            }
        }
        */
        for (int i = 0; i < _blockNum; i++)
        {
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], _kand_mask16(_inputMaskBlocks[i], radicalPosMaskBlock[i]),
                                                                _positiveLiteralBlocks[i], ones);
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], _kand_mask16(_inputMaskBlocksInverse[i], conservativeNegMaskBlock[i]),
                                                                _positiveLiteralBlocks[i], negOnes);
            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], _kand_mask16(_inputMaskBlocks[i], conservativeNegMaskBlock[i]),
                                                                _negativeLiteralBlocks[i], negOnes);
            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], _kand_mask16(_inputMaskBlocksInverse[i], radicalPosMaskBlock[i]),
                                                                _negativeLiteralBlocks[i], ones);
        }
       
    }
    else
    {
        /*
        for(int i=0; i<_literalNum; i++)
        {
            _positiveLiterals[i] -= conservative[i];
            _negativeLiterals[i] -= !(radical[i]);      // Use reverse of radical to prevent correspondance.
        }
        */
        for (int i = 0; i < _blockNum; i++)
        {
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], conservativeNegMaskBlock[i], _positiveLiteralBlocks[i], negOnes);
            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], _knot_mask16(radicalPosMaskBlock[i]), _negativeLiteralBlocks[i], negOnes);
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
    
    /*
    for (int i = 0; i < _literalNum; i++)
    {
        _positiveLiterals[i] += (!_posInclusionMask[i])&&(!_inputMask[i]);
        _negativeLiterals[i] += (!_negInclusionMask[i])&&(_inputMask[i]);
    }
    */
    static __m512i                      ones = _mm512_set1_epi32(1);
    for (int i = 0; i < _blockNum; i++)
    {
        _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], _kand_mask16(_knot_mask16(_posInclusionMaskBlocks[i]), _inputMaskBlocksInverse[i]),
                                                                                        _positiveLiteralBlocks[i], ones);
        _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], _kand_mask16(_knot_mask16(_negInclusionMaskBlocks[i]), _inputMaskBlocks[i]),
                                                                                        _negativeLiteralBlocks[i], ones);
    }
    
    _isVoteDirty = false;
}

/*
/// @brief Import clause state vectors from outside.
/// @param targetModel State vector unpackaged by Automata.
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

/// @brief Pack and export clause state vectors to outside.
/// @return Current clause state vectors.
Clause::model Clause::exportModel()
{
    Clause::model result;
    result.no = _no;
    result.positiveLiteral = _positiveLiterals;
    result.negativeLiteral = _negativeLiterals;
    return result;
}
*/
