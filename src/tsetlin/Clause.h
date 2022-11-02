#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <assert.h>
#include "pcg_random.hpp"
using std::vector;

/// @brief This clause use integer as literal as default.
class Clause{
public:
    struct ClauseArgs
    {
        int     no;         // Unique tag of all clauses.
        int     inputSize;
        double  specificity;
    };
    struct model
    {
        int             no;
        vector<int> positiveLiterals;
        vector<int> negativeLiterals;
        model(){}
    };

private:
    const int               _no;
    const int               _literalNum;
    const int               _blockNum;
    const double            _s,_sInv,_sInvConj;     // Allocated granular.

    static const inline __m512i     _ones = _mm512_set1_epi32(1);
    static const inline __m512i     _zeros = _mm512_set1_epi32(0);
    static const inline __m512i     _negOnes= _mm512_set1_epi32(-1);
    static const inline __mmask16   _zeroMask = _mm512_cmpeq_epi32_mask(_ones,_zeros);
    static const inline __mmask16   _oneMask = _mm512_cmpeq_epi32_mask(_ones,_ones);

    pcg64_fast          _rng;
    vector<__m512i>     _positiveLiteralBlocks;
    vector<__m512i>     _negativeLiteralBlocks;
    vector<__mmask16>   _posInclusionMaskBlocks;
    vector<__mmask16>   _negInclusionMaskBlocks;
    vector<__mmask16>   _posExclusionMaskBlocks;
    vector<__mmask16>   _negExclusionMaskBlocks;
    vector<__mmask16>   _inputMaskBlocks;
    vector<__mmask16>   _inputMaskBlocksInverse;
    __mmask16           _lastValidMask;         // Boundary problem
    int                 _vote;
    bool                _isVoteDirty;

    bool                modelIntegrityCheck(model &targetModel);
    vector<int>         unpack(vector<__m512i> &original);
    vector<__m512i>     pack(vector<int> &original);
public:
    Clause(ClauseArgs args);
    int     vote(vector<__m512i> &in);
    void    feedbackTypeI();
    void    feedbackTypeII();

    model   exportModel();
    void    importModel(model &targetModel);
};