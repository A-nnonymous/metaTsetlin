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

#pragma once
#include <vector>
#include <string>
#include <string_view>
#include <initializer_list>
#include <functional>
#include "io.h"

using std::vector;
using std::string;

typedef std::function<vector<vector<int>>(const vector<string>&)> seqParseFunc;
typedef std::function<vector<string>(const vector<vector<int>>&, int, int)> seqDeparseFunc;// signal and offset.

///////////////////////////// Tsetlin machine specified maps /////////////////////////////

static std::unordered_map<char,vector<int>> dnaSeq2Vec = 
{
    {'A', vector<int>{1, 0, 0, 0}},
    {'a', vector<int>{1, 0, 0, 0}},
    {'T', vector<int>{0, 1, 0, 0}},
    {'t', vector<int>{0, 1, 0, 0}},
    {'C', vector<int>{0, 0, 1, 0}},
    {'c', vector<int>{0, 0, 1, 0}},
    {'G', vector<int>{0, 0, 0, 1}},
    {'g', vector<int>{0, 0, 0, 1}},
};

static std::unordered_map<char,vector<int>> rnaSeq2Vec = 
{
    {'A', vector<int>{1, 0, 0, 0}},
    {'a', vector<int>{1, 0, 0, 0}},
    {'U', vector<int>{0, 1, 0, 0}},
    {'u', vector<int>{0, 1, 0, 0}},
    {'C', vector<int>{0, 0, 1, 0}},
    {'c', vector<int>{0, 0, 1, 0}},
    {'G', vector<int>{0, 0, 0, 1}},
    {'g', vector<int>{0, 0, 0, 1}},
};
/// Used to interpret tsetlin literal as nucleotide.
static std::unordered_map<int,char> tag2CharPos= 
{
    {0, '_'},
    {1, 'A'},
    {2, 'U'},
    {4, 'C'},
    {8, 'G'},
};

static std::unordered_map<int,char> tag2CharNeg= 
{
    {0, '_'},
    {1, 'W'},
    {2, 'X'},
    {4, 'Y'},
    {8, 'Z'},
};

static vector<std::unordered_map<int,string>> feature2StrMap(1);
///////////////////////////// Tsetlin machine specified maps /////////////////////////////

vector<int> siRNA2SIG(const string &raw_string)noexcept;
vector<double> seqGC(const vector<string> &rawStrings)noexcept;

class Parser{
private:
    /////////////// Data shared to decoder////////////////
    friend class                Deparser;
    vector<int>                 workRange;
    vector<vector<double>>      discreteThreshold;
    int                         funcNum;
    /////////////// Data shared to decoder////////////////
    vector<seqParseFunc>        funcs;
    vector<vector<int>>         parseSeqs2NucSig(const vector<string> &rawStrings)const noexcept;
    vector<vector<int>>         parseSeqs2GCSig(const vector<string> &rawStrings)noexcept;

    vector<vector<int>> concatenate(vector<vector<vector<int>>> &signals)noexcept;
public:
    Parser();
    vector<vector<int>> parse(const vector<string> &rawStrings)noexcept;
};

class Deparser{
private:
    //////////////////Received from encoder////////////////////
    const vector<int>               &workRange;
    const vector<vector<double>>    &discreteThreshold; // initialized to be invalid until parse complete.
    //////////////////Received from encoder////////////////////
    int                             funcNum;
    vector<seqDeparseFunc>          funcs;
    vector<string>                  deparseNucSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)const noexcept;
    vector<string>                  deparseGCSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)const noexcept;

    vector<string>                  concatenate(const vector<vector<string>> &discriptions)const noexcept;

public:
    Deparser(const Parser &psr)noexcept;
    vector<string> deparse(const vector<vector<int>> &rawSignals)const noexcept;
};

class nucTransformer{
private:
    Parser parser;
    Deparser deparser = Deparser(parser);
    vector<double> responseThreshold;

public:
    nucTransformer()noexcept{};

    dataset parseAndDivide( const vector<string> &seqs,
                            const vector<double> &responses,
                            double trainRatio, int classes)noexcept;
    
    void deparseAndOutput(  const TsetlinMachine::model &trainedModel,
                            double precision,
                            vector<string> headers,
                            string outputPath)const noexcept;
};