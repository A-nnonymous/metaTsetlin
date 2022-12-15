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

    pcg64_fast              _rng;
    vector<__m512i>         _positiveLiteralBlocks;
    vector<__m512i>         _negativeLiteralBlocks;
    vector<__mmask16>       _posInclusionMaskBlocks;
    vector<__mmask16>       _negInclusionMaskBlocks;
    vector<__mmask16>       _posExclusionMaskBlocks;
    vector<__mmask16>       _negExclusionMaskBlocks;
    vector<__mmask16>       _inputMaskBlocks;
    vector<__mmask16>       _inputMaskBlocksInverse;
    __mmask16               _lastValidMask;         // Boundary problem
    
    int                     _vote;

    //bool                    modelIntegrityCheck(model &targetModel);
    vector<int>             unpack(vector<__m512i> &original)noexcept;
    vector<__m512i>         pack(vector<int> &original)noexcept;
public:
    Clause(ClauseArgs args)noexcept;

    int                     vote(vector<__m512i> &in)noexcept;
    void                    feedbackTypeI()noexcept;
    void                    feedbackTypeII()noexcept;

    vector<int>             exportModel();
    //void                    importModel(model &targetModel);
};