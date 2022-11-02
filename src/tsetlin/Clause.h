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
        bool    isPositiveClause;
        int     inputSize;
        double  specificity;
    };
    struct model
    {
        int         no;
        vector<int> positiveLiteral;
        vector<int> negativeLiteral;
        model(){}
    };

private:
    const int           _no;
    const bool          _isPositiveClause;
    const int           _literalNum;
    const int           _blockNum;
    const double        _s,_sInv,_sInvConj;     // Allocated granular.

    pcg64_fast          _rng;
    // May pack the integer to 16 per group.
    vector<__m512i>     _positiveLiteralBlocks;
    vector<__m512i>     _negativeLiteralBlocks;
    // May convert the inclusion mask and inputmask to __mmask16
    __mmask16           _lastValidMask;         // Boundary problem
    vector<__mmask16>   _posInclusionMaskBlocks;
    vector<__mmask16>   _negInclusionMaskBlocks;
    vector<__mmask16>   _posExclusionMaskBlocks;
    vector<__mmask16>   _negExclusionMaskBlocks;
    vector<__mmask16>   _inputMaskBlocks;
    vector<__mmask16>   _inputMaskBlocksInverse;
    int                 _vote;
    bool                _isVoteDirty;

    bool                modelIntegrityCheck(model targetModel);
    vector<int>         unpack(vector<__m512i> original);
    vector<__m512i>     pack(vector<int> original);
public:
    Clause(ClauseArgs args);

    int     vote(vector<__m512i> in);
    void    feedbackTypeI();
    void    feedbackTypeII();
};