#include "Clause.h"
using std::vector;
using std::cout, std::endl;

Clause::Clause(ClauseArgs args)noexcept:
_no(args.no),
_literalNum(args.inputSize),
_s(args.specificity), _sInv(1.0/_s), _sInvConj(1.0-_sInv),
_blockNum(args.inputSize/16 + (args.inputSize%16==0? 0:1))
{
    
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64_fast          _rng(seed_source);

    _positiveLiteralBlocks.resize(_blockNum, _zeros);
    _posInclusionMaskBlocks.resize(_blockNum, _zeroMask);
    _posExclusionMaskBlocks.resize(_blockNum, _zeroMask);

    _negativeLiteralBlocks.resize(_blockNum, _zeros);
    _negInclusionMaskBlocks.resize(_blockNum, _zeroMask);
    _negExclusionMaskBlocks.resize(_blockNum, _zeroMask);

    _inputMaskBlocks.resize(_blockNum, _zeroMask);
    _inputMaskBlocksInverse.resize(_blockNum, _zeroMask);

    // Compact memory usage.
    _positiveLiteralBlocks.shrink_to_fit();
    _posInclusionMaskBlocks.shrink_to_fit();
    _posExclusionMaskBlocks.shrink_to_fit();
    _negativeLiteralBlocks.shrink_to_fit();
    _negInclusionMaskBlocks.shrink_to_fit();
    _negExclusionMaskBlocks.shrink_to_fit();
    _inputMaskBlocks.shrink_to_fit();
    _inputMaskBlocksInverse.shrink_to_fit();
    
    int remainder = _literalNum%16;         // Deal with boundary problem.
    int lastMaskInt = 0;
    for (int offset = 0; offset < remainder; offset++)
    {
        lastMaskInt += (1<<offset);
    }
    _lastValidMask = _mm512_int2mask(lastMaskInt);

    _vote = 0;
}

/// @brief Pack and 'align' the original vector of int to 512Byte pack with zero-padding if size not equal to 16-mer.
/// @param original Original vector of 32bit integer.
/// @return Vector of packed and zero-padded __m512i pack vector.
vector<__m512i>
Clause::pack(vector<int> &original)noexcept
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
    result.shrink_to_fit();
    return result;
}

vector<int>
Clause::unpack(vector<__m512i> &original)noexcept
{
    alignas(64) struct pack{
        int data[16];
        pack(){
            for (int i = 0; i < 16; i++)
            {
                data[i] = 0;
            }
            
        }
    };
    vector<pack> res(_blockNum,pack());
    for (int i = 0; i < _blockNum; i++)
    {
        _mm512_storeu_epi32(&res[i].data, original[i]);
    }
    vector<int> result(_literalNum,0);
    for (int i = 0; i < _literalNum; i++)
    {
        result[i] = res[i/16].data[i%16];
    }
    
    return result;
}

/// @brief Check model integrity before importing
/// @param targetModel Model that user intend to import
/// @return Boolean value of the integrity
bool Clause::modelIntegrityCheck(model &targetModel)
{
    bool isRightLength =    targetModel.literals.size() == 2* _literalNum;
    bool isRightPlace = (targetModel.no == _no);
    return isRightLength && isRightPlace;
}

/// @brief Vote function used for both train and predict procedure.
/// @param in Data vector that in the shape of ( 1, _literalNum )
/// @return Vote result, 0 or 1.
int Clause::vote(vector<__m512i> &in)noexcept
{
    vector<__mmask16>   posWrong(_blockNum, _zeroMask);
    vector<__mmask16>   negWrong(_blockNum, _zeroMask);
    
    for (int i = 0; i < _blockNum; i++)           // Run once per data input.
    {
        _posInclusionMaskBlocks[i] = _mm512_cmpge_epi32_mask(_positiveLiteralBlocks[i], _zeros);
        _negInclusionMaskBlocks[i] = _mm512_cmpge_epi32_mask(_negativeLiteralBlocks[i], _zeros);
        _inputMaskBlocks[i] = _mm512_cmpgt_epi32_mask(in[i], _zeros);                // Greater than zero (only possible value is one)

        _posExclusionMaskBlocks[i] = _knot_mask16(_posInclusionMaskBlocks[i]);
        _negExclusionMaskBlocks[i] = _knot_mask16(_negInclusionMaskBlocks[i]);
        _inputMaskBlocksInverse[i] = _knot_mask16(_inputMaskBlocks[i]);
        
        if(i == (_blockNum-1))[[unlikely]]    // Last block, may have boundary problem.
        {
            _posExclusionMaskBlocks[i] = _kand_mask16(_posExclusionMaskBlocks[i], _lastValidMask);
            _negExclusionMaskBlocks[i] = _kand_mask16(_negExclusionMaskBlocks[i], _lastValidMask);
            _inputMaskBlocksInverse[i] = _kand_mask16(_inputMaskBlocksInverse[i], _lastValidMask);
        }
    }
    bool posNoProblem = true, negNoProblem= true;
    bool hasProblem;
    for (int i = 0; i < _blockNum; i++)
    {
        posWrong[i] =  _kor_mask16( posWrong[i],
                                    _kand_mask16(   _posInclusionMaskBlocks[i],
                                                    _inputMaskBlocksInverse[i]));   // Included but input=0, positive literal wrong.

        negWrong[i] =  _kor_mask16( negWrong[i],
                                    _kand_mask16(   _negInclusionMaskBlocks[i],
                                                    _inputMaskBlocks[i]));          // Included but input=1, negative literal wrong.
        posNoProblem = (_mm512_mask2int(posWrong[i]) == 0 );    // So far so good.
        negNoProblem = (_mm512_mask2int(negWrong[i]) == 0 );
        hasProblem = !(posNoProblem && negNoProblem);
        if(hasProblem)[[likely]] break;         // Break when first unsatisfied literal occured.
    }
    int result = (hasProblem? 0:1);
    _vote = result;
    return result;
}

/// @brief Reinforce positive and negative literals according to 's' ,input, previous vote.
void Clause::feedbackTypeI()noexcept
{
    std::discrete_distribution<> d({_sInv, _sInvConj});
    vector<int>                  radical(_literalNum,0);
    vector<int>                  conservative(_literalNum,0);
    vector<__m512i>              radicalBlock(_blockNum, _zeros);
    vector<__m512i>              conservativeBlock(_blockNum, _zeros);
    vector<__mmask16>            radicalPosMaskBlock(_blockNum, _zeroMask);
    vector<__mmask16>            conservativeNegMaskBlock(_blockNum, _zeroMask);
    vector<__mmask16>            conservativeNegMaskBlock2(_blockNum, _zeroMask);

    for (int i = 0; i < _literalNum; i++)
    {
        radical[i] =        (d(_rng)==1);   // Fill 'true' with possibility of _sInvConj
        conservative[i] =   (d(_rng)==0);   // Complement possibility, but not correlated to radical[].
    }
    radicalBlock = pack(radical); conservativeBlock = pack(conservative);

    for (int i = 0; i < _blockNum; i++)
    {
        radicalPosMaskBlock[i] = _mm512_cmpeq_epi32_mask(radicalBlock[i], _ones);
        conservativeNegMaskBlock[i] = _mm512_cmpeq_epi32_mask(conservativeBlock[i], _ones);
        conservativeNegMaskBlock2[i] = _knot_mask16(radicalPosMaskBlock[i]);
        if(i == (_blockNum-1))[[unlikely]]
        {
            conservativeNegMaskBlock2[i] = _kand_mask16(conservativeNegMaskBlock2[i], _lastValidMask);
        }
    }
    if(_vote)
    {
        for (int i = 0; i < _blockNum; i++)
        {
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i],
                                                                _kand_mask16(   _inputMaskBlocks[i],
                                                                                radicalPosMaskBlock[i]),
                                                                _positiveLiteralBlocks[i], _ones);

            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i],
                                                                _kand_mask16(   _inputMaskBlocksInverse[i],
                                                                                conservativeNegMaskBlock[i]),
                                                                _positiveLiteralBlocks[i], _negOnes);

            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i],
                                                                _kand_mask16(   _inputMaskBlocks[i],
                                                                                conservativeNegMaskBlock[i]),
                                                                _negativeLiteralBlocks[i], _negOnes);

            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i],
                                                                _kand_mask16(   _inputMaskBlocksInverse[i],
                                                                                radicalPosMaskBlock[i]),
                                                                _negativeLiteralBlocks[i], _ones);
        }
    }
    else
    {
        for (int i = 0; i < _blockNum; i++)
        {
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i],
                                                                conservativeNegMaskBlock[i],
                                                                _positiveLiteralBlocks[i], _negOnes);

            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i],
                                                                conservativeNegMaskBlock2[i],
                                                                _negativeLiteralBlocks[i], _negOnes);
        }
    }
}


/// @brief Reinforce positive and negative literals according to inclusion, input, previous vote.
/// @param in Previous input vector.
void Clause::feedbackTypeII()noexcept
{
    if(_vote==0)return;
    for (int i = 0; i < _blockNum; i++)
    {
        _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i],
                                                            _kand_mask16(   _posExclusionMaskBlocks[i],
                                                                            _inputMaskBlocksInverse[i]),
                                                            _positiveLiteralBlocks[i], _ones);

        _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], 
                                                            _kand_mask16(   _negExclusionMaskBlocks[i],
                                                                            _inputMaskBlocks[i]),
                                                            _negativeLiteralBlocks[i], _ones);
    }
}

Clause::model Clause::exportModel()
{
    model result;
    result.no = _no;
    vector<int> posLitFilled = unpack(_positiveLiteralBlocks);
    vector<int> negLitFilled = unpack(_negativeLiteralBlocks);
    vector<int> literals(_literalNum * 2, 0);
    for (int i = 0; i < _literalNum; i++)
    {
        literals[i] = posLitFilled[i];
        literals[i+_literalNum] = negLitFilled[i];
    }
    result.literals = literals;
    return result;
}

void Clause::importModel(model &targetModel)
{
    if(!Clause::modelIntegrityCheck(targetModel))[[unlikely]]
    {
        std::cout<<"Your Tsetlin Machine model failed integrity check!"<<std::endl;
        throw; return;
    }
    vector<int> pos(_literalNum, 0);
    vector<int> neg(_literalNum, 0);
    for (int i = 0; i < _literalNum; i++)
    {
        pos[i] = targetModel.literals[i];
        neg[i] = targetModel.literals[i+_literalNum];
    }
    
    _positiveLiteralBlocks = pack(pos);
    _negativeLiteralBlocks = pack(neg);
}