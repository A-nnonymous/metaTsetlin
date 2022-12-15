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
v
#include "nucleotides.h"
using std::string_view;

/// @brief Encode SiRNA sequence to 4-bit integer.
/// @param raw raw SiRNA sequences
/// @return Vector of encoded SiRNA.
vector<int> siRNA2SIG(const string &raw)noexcept
{
    vector<int> result;
    result.reserve(raw.size() * sizeof(int));   // Pre-allocate space to avoid logarithmic re-allocation.
    auto sv = string_view(raw);
    char ch;
    size_t base = 0;
    short offset = 0;
    for(size_t idx = 0; idx < sv.size(); idx++)
    {
        ch = sv.at(idx);
        auto code = dnaSeq2Vec[ch].empty()? rnaSeq2Vec[ch] : dnaSeq2Vec[ch];
        result.insert(result.end(), code.begin(), code.end());
    }
    result.shrink_to_fit();
    return result;
}

/// @brief Compute all sequences' GC content.
/// @param rawStrings Vector of all nucleotide sequences.
/// @return Vector of GC contents corresponding to all sequences.
vector<double> seq2GC(const vector<string> &raw)noexcept
{
    string_view sv;
    auto sampleNum = raw.size();
    vector<double> result(sampleNum, 0);
    for(auto idx = 0; idx < sampleNum; idx++)
    {
        sv = string_view(raw[idx]);
        result[idx] = ( std::count(sv.begin(), sv.end(),'G') +
                        std::count(sv.begin(), sv.end(),'C') ) / (double)sv.size();
    }
    result.shrink_to_fit();
    return result;
}

/////////////////////////////// Parser member functions //////////////////////////////////////////
Parser::Parser()
{
    //////////// Extensible part (but forgive my ugly code, sry)//////////////
    seqParseFunc Nuc =
    [&](const vector<string> &rawStrings)
    -> vector<vector<int>> 
    {
        return this->parseSeqs2NucSig(rawStrings);
    };

    seqParseFunc GC =
    [&](const vector<string> &rawStrings)
    -> vector<vector<int>> 
    {
        return this->parseSeqs2GCSig(rawStrings);   // Modification of threshold vector.
    };

    funcs.emplace_back(Nuc);
    funcs.emplace_back(GC);
    //////////// Extensible part (but forgive my ugly code, sry)//////////////
    
    funcNum = funcs.size();
    workRange.resize(funcNum + 1, 0);
    discreteThreshold.resize(funcNum - 1, vector<double>(4,0)); // Nucleotide transform need no threshold.
}

/// @brief Concatenate different parsers's output into train-ready 2d-vector shaped in SampleNum * featureNum.
/// @param signals Output from various parser.
/// @return Train-ready vector.
vector<vector<int>> Parser::concatenate(vector<vector<vector<int>>> &signals)noexcept
{
    auto sampleLength = 0;
    auto sampleNum = signals[0].size(); // Assuming no discard of samples in all parser.
    vector<vector<int>> result(sampleNum);
    // Adding up all feature`s length.
    for(auto i = 0 ; i < funcNum; i++)
    {
        sampleLength += signals[i][0].size();
        workRange[i + 1] = sampleLength; // Modification of shared class member value "workRange"
    }
    // Reserve and put all generated feature in a new 2d vector
    for (auto sampleIdx = 0; sampleIdx < sampleNum; sampleIdx++)
    {
        result[sampleIdx].reserve(sampleLength);
        for(auto funcIdx = 0; funcIdx < funcNum; funcIdx++)
        {
            result[sampleIdx].insert(   result[sampleIdx].end(),
                                        signals[funcIdx][sampleIdx].begin(),
                                        signals[funcIdx][sampleIdx].end());
        }
    }
    return result;
}

/// @brief Parse the original nucleotide into feature vectors
/// @param rawStrings Vector of SiRNA nucleotide strings.
/// @return Train-ready 2D vector shaped in SampleNum * FeatureNum.
vector<vector<int>> Parser::parse(const vector<string> &rawStrings)noexcept
{
    auto funcSize = funcs.size();
    vector<vector<int>> result;
    vector<vector<vector<int>>> seperateResult(funcSize);
    for(auto fIdx = 0; fIdx < funcSize; fIdx++)
    {
        seperateResult[fIdx] = funcs[fIdx](rawStrings); // Modification of threshold vector.
    }
    result = this->concatenate(seperateResult);
    return result;
}

/// @brief Sequence parse function: encode nucleotide string into vector of integer.
/// @param rawStrings Original vector of nucleotide string.
/// @return Train-ready 2D-vector.
vector<vector<int>> Parser::parseSeqs2NucSig(const vector<string> &rawStrings)const noexcept
{
    int sampleNum = rawStrings.size();
    vector<vector<int>> result(sampleNum);
    for (auto i = 0; i < sampleNum; i++)
    {
        result[i] = siRNA2SIG(rawStrings[i]);
    }
    return result;
}

/// @brief Sequence parse function: evaluate and encode nucleotide strings' GC content ratio discretely.
/// @param rawStrings Original vector of nucleotide string.
/// @return Train-ready 2D-vector.
vector<vector<int>> Parser::parseSeqs2GCSig(const vector<string> &rawStrings)noexcept
{
    const int funcIdx = 0;
    const int thresholdNum = 8;
    int sampleNum = rawStrings.size();
    vector<vector<int>> result(sampleNum);

    auto continuousResult = seq2GC(rawStrings);
    auto threshold = getFairThreshold<double>(continuousResult, thresholdNum);
    discreteThreshold[funcIdx] = threshold;
    for (auto i = 0; i < sampleNum; i++)
    {
        result[i] = getDiscreteResponse(threshold, continuousResult[i]);
    }
    return result;
}
/////////////////////////////// Parser member functions //////////////////////////////////////////


/////////////////////////////// Deparser member functions //////////////////////////////////////////

/// @brief Initialize Deparser using corresponding parser.
/// @param psr Corresponding parser, containing essential informations.
Deparser::Deparser(const Parser &psr)noexcept:
workRange(psr.workRange), discreteThreshold(psr.discreteThreshold)
{
    seqDeparseFunc Nuc =
    [&](const vector<vector<int>> &rawSignals, const int start, const int end)
    -> vector<string>
    {
        auto result = this->deparseNucSig2Seq(rawSignals, start, end);
        return result;
    };
    seqDeparseFunc GC =
    [&](const vector<vector<int>> &rawSignals, const int start, const int end)
    -> vector<string>
    {
        auto result = this->deparseGCSig2Seq(rawSignals, start, end);
        return result;
    };
    funcs.emplace_back(Nuc);
    funcs.emplace_back(GC);
    funcNum = funcs.size();
}

/// @brief Deparse specified subset of trained tsetlin model clause(represented in integers) into nucleotides.
/// @param rawSignals Trained Tsetlin Machine model.
/// @param start Start index of nucleotide literals in each clause.
/// @param end End index of nucleotide literals in each clause.
/// @return Description of nucleotide in/exclusion pattern, containing pattern string and it's value.
vector<string> Deparser::deparseNucSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)const noexcept
{
    auto clauseNum = rawSignals.size();
    auto posPartLen = rawSignals[0].size()/2; // Represent tsetlin positive literal number.
    vector<string> result(clauseNum);
    for (auto clauseIdx = 0; clauseIdx < clauseNum; clauseIdx++)
    {
        string thisSamplePattern;
        bool isValidClause=true;
        auto validLiteral = 0;
        auto voice = 0;
        for (int baseIdx = start; baseIdx < end; baseIdx+=4)
        {
            int pValue, nValue;
            bool pActivate, nActivate;
            int pFlag = 0, nFlag = 0;
            int pForce = 0, nForce = 0;     // represent literal force if bipolar-literal not contradict and both exist.

            for(int offset = 0; offset < 4; offset++)
            {
                pValue = rawSignals[clauseIdx][baseIdx + offset];
                pActivate = pValue>=0;
                pForce =  pActivate ? pValue : pForce; // stay the same or be valid.
                pFlag += (1 << offset) * (pActivate);

                nValue= rawSignals[clauseIdx][baseIdx + offset + posPartLen];
                nActivate = nValue>=0;
                nForce = nActivate? nValue : nForce;
                nFlag += (1 << offset) * (nActivate);

                voice += abs(pValue) + abs(nValue);
            }

            auto chP = tag2CharPos[pFlag], chN = tag2CharNeg[nFlag];
            bool containsNull = (!chP || !chN);
            bool containsContradict = (pFlag == nFlag) && (pFlag != 0);

            if(containsNull || containsContradict)
            {
                //std::cout<< "failed in base "<< baseIdx<< "with pflag as: "<< pFlag << ", nflag as: "<< nFlag<<std::endl;
                isValidClause = false;
                break;
            }
            thisSamplePattern += (pForce > nForce)? chP : chN;
            validLiteral++;
        }
        double value = (validLiteral / (double)(posPartLen / (double)2) * voice); // valid literal ratio multiply to voice.
        if(isValidClause)result[clauseIdx] = thisSamplePattern + "," + std::to_string(value);
    }
    return result;
}

/// @brief Deparse specified subset of trained tsetlin model clause(represented in integers) into GC content.
/// @param rawSignals Trained Tsetlin Machine model.
/// @param start Start index of continuous GC content literals in each clause.
/// @param end End index of continuous GC content literals in each clause.
/// @return Description of GC content tier pattern.
vector<string> Deparser::deparseGCSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)const noexcept
{
    std::unordered_map<int, string> gcMapPos, gcMapNeg;
    auto clauseNum = rawSignals.size();
    auto posPartLen = rawSignals[0].size()/2; // Represent tsetlin positive literal number.
    vector<string> result(clauseNum);

    auto posTag = threshold2Tags(discreteThreshold[0], true);
    gcMapPos.insert({0,posTag[0]});
    gcMapNeg.insert({0,"Not " + posTag[0]});
    for (int i = 0; i < posTag.size() - 1; i++)
    {
        gcMapPos.insert({1<<i,posTag[i + 1]});
        gcMapNeg.insert({1<<i,"Not " + posTag[i + 1]});
    }
    
    
    for (auto clauseIdx = 0; clauseIdx < clauseNum; clauseIdx++)
    {
        bool isValidClause = true;
        string thisSamplePattern;
        auto validLiteral = 0;
        auto voice = 0;
        for (int baseIdx = start; baseIdx < end; baseIdx+=8)
        {
            int pValue, nValue;
            bool pActivate, nActivate;
            int pFlag = 0, nFlag = 0;
            int pForce = 0, nForce = 0;     // represent literal force if bipolar-literal not contradict and both exist.

            for(int offset = 0; offset < 4; offset++)
            {
                pValue = rawSignals[clauseIdx][baseIdx + offset];
                pActivate = pValue >= 0;
                pForce =  pActivate ? pValue : pForce; // stay the same or be valid.
                pFlag += (1 << offset) * (pActivate);

                nValue= rawSignals[clauseIdx][baseIdx + offset + posPartLen];
                nActivate = nValue >= 0;
                nForce = nActivate? nValue : nForce;
                nFlag += (1 << offset) * (nActivate);

                voice += abs(pValue) + abs(nValue);
            }

            auto chP = gcMapPos[pFlag], chN = gcMapNeg[nFlag];
            auto containsNull = (chP.empty() || chN.empty());
            auto containsContradict = (pFlag == nFlag) && (pFlag != 0);

            if(containsNull || containsContradict)
            {
                isValidClause = false;
                break;
            }
            thisSamplePattern += (pForce > nForce)? chP : chN;
            validLiteral++;
        }
        double value = (validLiteral / (double)(posPartLen / (double)2) * voice); // valid literal ratio multiply to voice.
        if(isValidClause)result[clauseIdx] = thisSamplePattern + "," + std::to_string(value);
    }
    return result;
}

/// @brief Concatenate output of various deparser into a integrated string.
/// @param discriptions Output of various deparser.
/// @return Human readable strings containing all deparsers' valid outputs.
vector<string> Deparser::concatenate(const vector<vector<string>> &discriptions)const noexcept
{
    auto clauseNum= discriptions[0].size();    // Assuming always exist a discription of all samples.
    vector<string> result;
    result.reserve(clauseNum);
    // Adding up all feature`s length.
    for (auto clauseIdx = 0; clauseIdx< clauseNum; clauseIdx++)
    {
        string thisClause;
        bool isValidClause = true;
        for(auto funcIdx = 0; funcIdx < funcNum; funcIdx++)
        {
            if(discriptions[funcIdx][clauseIdx].empty())
            {
                isValidClause = false;
                break;
            }
            else
            {
                thisClause += discriptions[funcIdx][clauseIdx];
                if(funcIdx != funcNum - 1) thisClause += ",";
            }
        }
        if(isValidClause) result.emplace_back(thisClause);
    }
    result.shrink_to_fit();
    return result;
}

/// @brief Deparse Tsetlin machine's model into human readable discriptions.
/// @param rawSignals A batch of Tsetlin clauses with same polarity.
/// @return Human readable strings containing all deparsers' valid outputs.
vector<string> Deparser::deparse(const vector<vector<int>> &rawSignals)const noexcept
{
    vector<string> result;
    vector<vector<string>> seperateResult(funcNum);
    for(auto fIdx = 0; fIdx < funcNum; fIdx++)
    {
        seperateResult[fIdx] = funcs[fIdx](rawSignals, workRange[fIdx], workRange[fIdx + 1]);
    }
    result = this->concatenate(seperateResult);
    return result;
}
/////////////////////////////// Deparser member functions //////////////////////////////////////////


//////////////////////////////////// NucTransformer member functions//////////////////////////////////

/// @brief Parse vector of nucleotide string into feature vector of integer and random pick in given ratio.
/// @param seqs Nucleotide strings loaded in memory.
/// @param responses Expected silence efficiency(target output).
/// @param trainRatio The ratio of training set with respect to whole data.
/// @param classes Number of discrete classes to split the responses.
/// @return Train-ready data structure.
dataset nucTransformer::parseAndDivide( const vector<string> &seqs,
                        const vector<double> &responses,
                        double trainRatio, int classes)noexcept
{
    ////////// arg check ////////////
    bool isRightLength = seqs.size() == responses.size();
    bool isPossibleDivision = classes <= seqs.size();
    if(!isRightLength)[[unlikely]] std::cout<<"Data length corrupted."<<std::endl;
    if(!isPossibleDivision)[[unlikely]] std::cout<<"Can't divide data into "<<classes<<" parts."<<std::endl;
    if(!isRightLength || !isPossibleDivision)[[unlikely]] return dataset();
    ////////// arg check ////////////
    
    auto totalData = parser.parse(seqs);

    ////////// Response threshold picking////////////
    responseThreshold = getFairThreshold<double> (responses, classes);
    ////////// Response threshold picking////////////

    ////////// Sequence prepare and random chosing//////////////
    std::discrete_distribution<>    d({1-trainRatio, trainRatio});
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64_fast                      _rng(seed_source);
    dataset result;

    vector<vector<int>>     trainData;
    vector<vector<int>>     trainResponse;
    vector<vector<int>>     testData;
    vector<vector<int>>     testResponse;

    vector<int> thisResponse;
    for (int i = 0; i < seqs.size(); i++)
    {
        thisResponse = getDiscreteResponse(responseThreshold, responses[i]);
        if(d(_rng)==1)[[likely]]    // Picked to training set.
        {
            trainData.emplace_back(totalData[i]);
            trainResponse.emplace_back(thisResponse);
        }
        else[[unlikely]]            // Picked to test set.
        {
            testData.emplace_back(totalData[i]);
            testResponse.emplace_back(thisResponse);
        }
    }
    trainData.shrink_to_fit();
    trainResponse.shrink_to_fit();
    testData.shrink_to_fit();
    testResponse.shrink_to_fit();
    ////////// Sequence prepare and random chosing//////////////
    
    result.responseSize = classes;
    result.responseThreshold = responseThreshold;
    result.trainData = trainData;
    result.trainResponse = trainResponse;
    result.testData = testData;
    result.testResponse = testResponse;
    result.trainSize = trainData.size();
    result.testSize = testData.size();
    result.tierTags = threshold2Tags(responseThreshold,true);

    //////////////////// verbose////////////////////
    std::cout<< "Dataset preparation completed."<<std::endl;
    std::cout<<"Actual training set ratio is "<< trainData.size()/(double)seqs.size()<<std::endl;
    std::cout<<"Balanced threshold for discrete response is:\t";
    for (auto i = 0; i < responseThreshold.size(); i++)
    {
        std::cout<< responseThreshold[i]<<"\t";
    }
    std::cout<<std::endl;
    //////////////////// verbose////////////////////
    return result;
}

/// @brief Deparse the Tsetlin model and output human readable knowledges.
/// @param trainedModel Trained tsetlin machine model.
/// @param precision Precision of given trained model.
/// @param headers Headers of output CSV.
/// @param outputPath Output path.
void nucTransformer::deparseAndOutput(  const TsetlinMachine::model &trainedModel,
                        double precision,
                        vector<string> headers,
                        string outputPath)const noexcept
{
    if(!outputPath.ends_with("/"))outputPath+="/";
    std::filesystem::create_directories(outputPath);

    auto tierNum = trainedModel.tierTags.size();
    vector<vector<string>> positivePatterns(tierNum);
    vector<vector<string>> negativePatterns(tierNum);
    for(auto tierIdx = 0; tierIdx < tierNum; tierIdx++)
    {
        positivePatterns[tierIdx] = deparser.deparse(trainedModel.automatas[tierIdx].positiveClauses);
        negativePatterns[tierIdx] = deparser.deparse(trainedModel.automatas[tierIdx].negativeClauses);
        string prefix = outputPath + "/prec" + std::to_string(precision) + "/" + trainedModel.tierTags[tierIdx] +"/";
        std::filesystem::create_directories(prefix);
        std::ofstream positive(prefix + "positivePatterns.csv");
        std::ofstream negative(prefix + "negativePatterns.csv");
        for (int i = 0; i < headers.size(); i++)
        {
            positive<<headers[i]<<((i==headers.size()-1)? "\n": ",");
        }
        for(auto i=0; i < positivePatterns[tierIdx].size(); i++)
        {
            positive << positivePatterns[tierIdx][i];
            if(i != positivePatterns[tierIdx].size() - 1)[[likely]]
            {
                positive<< "\n";
            }
        }
        for (int i = 0; i < headers.size(); i++)
        {
            negative<<headers[i]<<((i==headers.size()-1)? "\n": ",");
        }
        for(auto i=0; i < negativePatterns[tierIdx].size(); i++)
        {
            negative << negativePatterns[tierIdx][i];
            if(i != negativePatterns[tierIdx].size() - 1)[[likely]]
            {
                negative << "\n";
            }
        }
    }
}
//////////////////////////////////// NucTransformer member functions//////////////////////////////////