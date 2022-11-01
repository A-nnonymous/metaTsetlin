#include "Clause.h"
using std::vector;
using std::cout, std::endl;

void Clause::dataProbe(__m512i data)
{
    alignas(64) struct pack
    {
        int data[16];
    };
    pack temp;
    _mm512_storeu_epi32(&temp.data, data);
    for (int j = 0; j < 16; j++)
    {
        std::cout<< temp.data[j];
        if ((j+1)%4 == 0 && j!=15)
        {
            std::cout<<"_";
        }
    }
        std::cout<<std::endl;
}
void Clause::dataProbe(vector<__m512i> data)
{
    alignas(64) struct pack
    {
        int data[16];
    };
    pack temp;
    for (int i = 0; i < data.size(); i++)
    {
        _mm512_storeu_epi32(&temp.data, data[i]);
        for (int j = 0; j < 16; j++)
        {
            std::cout<< temp.data[j];
            if ((j+1)%4 == 0 && j!=15)
            {
                std::cout<<"_";
            }
        }
        std::cout<<"\t";
    }
    std::cout<<"\n\n";
}

void Clause::maskProbe(__mmask16 mask)
{
    int result;
    result = _mm512_mask2int(mask);
        for (int bit = 0; bit <16; bit++)
        {
            std::cout<< ((result>>bit) %2 == 1);;
            if((bit+1)%4==0&&bit!=15)std::cout<<"_";
        }
        std::cout<<std::endl;
}

void Clause::maskProbe(vector<__mmask16> mask)
{
    vector<int> result(mask.size(),0);
    for (int i = 0; i < mask.size(); i++)
    {
        result[i] = _mm512_mask2int(mask[i]);
    }
    for (int i = 0; i < result.size(); i++)
    {
        for (int bit = 0; bit <16; bit++)
        {
            //std::cout<< ((result[i]>>bit) %2 == 1);
            std::cout<< ((result[i]>>bit) %2 == 1);;
            if((bit+1)%4==0&&bit!=15)std::cout<<"_";
        }
        std::cout<<"\t";
    }
    std::cout<<"\n\n";
}

void tag(std::string str)
{
    cout<<str<<endl;
}

Clause::Clause(ClauseArgs args):
_no(args.no),
_isPositiveClause(args.isPositiveClause),
_literalNum(args.inputSize),
_s(args.specificity), _sInv(1.0/_s), _sInvConj(1.0-_sInv),
_blockNum(args.inputSize/16 + (args.inputSize%16==0? 0:1))
{
    __m512i     ones = _mm512_set1_epi32(1);
    __m512i     zeros = _mm512_set1_epi32(0);
    __m512i     negOnes = _mm512_set1_epi32(-1);
    __mmask16   zeroMask = _mm512_cmpeq_epi32_mask(ones,zeros);
    __mmask16   oneMask = _mm512_cmpeq_epi32_mask(ones,ones);

    _rng = std::mt19937(std::random_device{}());
    _vote = 0;  _isVoteDirty = false;
    _positiveLiteralBlocks.resize(_blockNum, zeros);
    _negativeLiteralBlocks.resize(_blockNum, zeros);
    _posInclusionMaskBlocks.resize(_blockNum, zeroMask);
    _negInclusionMaskBlocks.resize(_blockNum, zeroMask);
    _posExclusionMaskBlocks.resize(_blockNum, zeroMask);
    _negExclusionMaskBlocks.resize(_blockNum, zeroMask);
    _inputMaskBlocks.resize(_blockNum, zeroMask);
    _inputMaskBlocksInverse.resize(_blockNum, zeroMask);

    int remainder = _literalNum%16;         // Deal with boundary problem.
    int lastMaskInt = 0;
    for (int offset = 0; offset < remainder; offset++)
    {
        lastMaskInt += (1<<offset);
    }
    _lastValidMask = _mm512_int2mask(lastMaskInt);
}




/// @brief Pack and 'align' the original vector of int to 512Byte pack with zero-padding if size not equal to 16-mer.
/// @param original Original vector of 32bit integer.
/// @return Vector of packed and zero-padded __m512i pack vector.
vector<__m512i>
Clause::pack(vector<int> original)
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
        _posInclusionMaskBlocks[i] = _mm512_cmpgt_epi32_mask(_positiveLiteralBlocks[i], zeros);
        _negInclusionMaskBlocks[i] = _mm512_cmpgt_epi32_mask(_negativeLiteralBlocks[i], zeros);
        _inputMaskBlocks[i] = _mm512_cmpgt_epi32_mask(in[i], zeros);                // Greater than zero (only possible value is one)

        _posExclusionMaskBlocks[i] = _knot_mask16(_posInclusionMaskBlocks[i]);
        _negExclusionMaskBlocks[i] = _knot_mask16(_negInclusionMaskBlocks[i]);
        _inputMaskBlocksInverse[i] = _knot_mask16(_inputMaskBlocks[i]);
        
        if(i == (_blockNum-1))    // Last block, may have boundary problem.
        {
            _posExclusionMaskBlocks[i] = _kand_mask16(_posExclusionMaskBlocks[i], _lastValidMask);
            _negExclusionMaskBlocks[i] = _kand_mask16(_negExclusionMaskBlocks[i], _lastValidMask);
            _inputMaskBlocksInverse[i] = _kand_mask16(_inputMaskBlocksInverse[i], _lastValidMask);
        }
    }
    /*
    //////////////////debug/////////////////////
    tag("Inputmask:");
    maskProbe(_inputMaskBlocks);
    tag("InputmaskInverse:");
    maskProbe(_inputMaskBlocksInverse);
    tag("posInclusionMask");
    maskProbe(_posInclusionMaskBlocks);
    tag("posExclusionMask");
    maskProbe(_posExclusionMaskBlocks);
    tag("negInclusionMask");
    maskProbe(_negInclusionMaskBlocks);
    tag("negExclusionMask");
    maskProbe(_negExclusionMaskBlocks);
    //////////////////debug/////////////////////
    */
    bool posNoProblem = true, negNoProblem= true;
    bool hasProblem;
    for (int i = 0; i < _blockNum; i++)
    {
        posWrong[i] =  _kor_mask16(posWrong[i],
                                                _kand_mask16(_posInclusionMaskBlocks[i],_inputMaskBlocksInverse[i]));    // Included but input=0, positive literal wrong.

        negWrong[i] =  _kor_mask16(negWrong[i],
                                                _kand_mask16(_negInclusionMaskBlocks[i],_inputMaskBlocks[i]));                  // Included but input=1, negative literal wrong.
        posNoProblem = (_mm512_mask2int(posWrong[i]) == 0 );    // So far so good.
        negNoProblem = (_mm512_mask2int(negWrong[i]) == 0 );
        hasProblem = !(posNoProblem && negNoProblem);
        if(hasProblem) break;         // Break when first unsatisfied literal occured.
    }
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
    static std::mt19937                 rng(std::random_device{}());
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
    static vector<__mmask16>            conservativeNegMaskBlock2(_blockNum, zeroMask);

    for (int i = 0; i < _literalNum; i++)
    {
        radical[i] =        (d(rng)==1);   // Fill 'true' with possibility of _sInvConj
        conservative[i] =   (d(rng)==0);   // Complement possibility, but not correlated to radical[].
    }
    radicalBlock = pack(radical); conservativeBlock = pack(conservative);

    for (int i = 0; i < _blockNum; i++)
    {
        radicalPosMaskBlock[i] = _mm512_cmpeq_epi32_mask(radicalBlock[i], ones);
        conservativeNegMaskBlock[i] = _mm512_cmpeq_epi32_mask(conservativeBlock[i], ones);
        conservativeNegMaskBlock2[i] = _knot_mask16(radicalPosMaskBlock[i]);
        if(i == (_blockNum-1))
        {
            conservativeNegMaskBlock2[i] = _kand_mask16(conservativeNegMaskBlock2[i], _lastValidMask);
        }
    }
    /*
    ///////////////debug//////////////
    {
    std::cout<<"radical judge array"<<std::endl;
    dataProbe(radicalBlock);
    std::cout<<"Conservative judge array"<<std::endl;
    dataProbe(conservativeBlock);
    std::cout<<"Conservative judge array2"<<std::endl;
    dataProbe(conservativeBlock2);
    std::cout<<"radical judge mask"<<std::endl;
    maskProbe(radicalPosMaskBlock);
    std::cout<<"Conservative judge mask"<<std::endl;
    maskProbe(conservativeNegMaskBlock);
    std::cout<<"Conservative judge mask2"<<std::endl;
    maskProbe(conservativeNegMaskBlock2);
    }
    ///////////////debug//////////////
    */
    if(_vote)
    {
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
        
        /*
        ////////////debug///////////
        {
            cout<<"voted 1"<<endl;
            cout<< "poslit after feedbackI"<<endl;
            dataProbe(_positiveLiteralBlocks);
            cout<< "neglit after feedbackI"<<endl;
            dataProbe(_negativeLiteralBlocks);
        }
        ////////////debug///////////
        */
    }
    else
    {
        for (int i = 0; i < _blockNum; i++)
        {
            _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], conservativeNegMaskBlock[i], _positiveLiteralBlocks[i], negOnes);
            _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], conservativeNegMaskBlock2[i], _negativeLiteralBlocks[i], negOnes);
        }
        /*
        ////////////debug///////////
        {
            cout<<"voted 0"<<endl;
            cout<< "poslit after feedbackI"<<endl;
            dataProbe(_positiveLiteralBlocks);
            cout<< "neglit after feedbackI"<<endl;
            dataProbe(_negativeLiteralBlocks);
        }
        ////////////debug///////////
        */
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
        _positiveLiteralBlocks[i] = _mm512_mask_add_epi32(  _positiveLiteralBlocks[i], _kand_mask16(_posExclusionMaskBlocks[i], _inputMaskBlocksInverse[i]),
                                                                                        _positiveLiteralBlocks[i], ones);
        _negativeLiteralBlocks[i] = _mm512_mask_add_epi32(  _negativeLiteralBlocks[i], _kand_mask16(_negExclusionMaskBlocks[i], _inputMaskBlocks[i]),
                                                                                        _negativeLiteralBlocks[i], ones);
    }
    
    /*
    ////////////debug/////////////////
    {
        cout<<"poslit after feedbackII:"<<endl;
        dataProbe(_positiveLiteralBlocks);
        cout<<"neglit after feedbackII:"<<endl;
        dataProbe(_negativeLiteralBlocks);
    }
    ////////////debug/////////////////

    */
}

