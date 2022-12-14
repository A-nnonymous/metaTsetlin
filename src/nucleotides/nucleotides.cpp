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

vector<vector<int>> Parser::parseSeqs2NucSig(const vector<string> &rawStrings)
{
    int sampleNum = rawStrings.size();
    vector<vector<int>> result(sampleNum);
    for (auto i = 0; i < sampleNum; i++)
    {
        result[i] = siRNA2SIG(rawStrings[i]);
    }
    return result;
}

vector<vector<int>> Parser::parseSeqs2GCSig(const vector<string> &rawStrings)
{
    const int funcIdx = 0;
    int sampleNum = rawStrings.size();
    vector<vector<int>> result(sampleNum);

    auto continuousResult = seq2GC(rawStrings);
    auto threshold = getFairThreshold<double>(continuousResult, 4);
    discreteThreshold[funcIdx] = threshold;
    for (auto i = 0; i < sampleNum; i++)
    {
        result[i] = getDiscreteResponse(threshold, continuousResult[i]);
    }
    return result;
}

/// @brief Deparse specified subset of trained tsetlin model clause(represented in integers) into nucleotides.
/// @param rawSignals Trained Tsetlin Machine model.
/// @param start Start index of nucleotide literals in each clause.
/// @param end End index of nucleotide literals in each clause.
/// @return Description of nucleotide in/exclusion pattern, containing pattern string and it's value.
vector<string> Deparser::deparseNucSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)
{
    auto sampleNum = rawSignals.size();
    auto posPartLen = rawSignals[0].size()/2; // Represent tsetlin positive literal number.
    vector<string> result(sampleNum);
    for (auto sampleIdx = 0; sampleIdx < sampleNum; sampleIdx++)
    {
        string thisSamplePattern;
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
                pValue = rawSignals[sampleIdx][baseIdx + offset];
                pActivate = !!pValue;
                pForce =  pActivate ? pValue : pForce; // stay the same or be valid.
                pFlag += (1 << offset) * (pActivate);

                nValue= rawSignals[sampleIdx][baseIdx + offset + posPartLen];
                nActivate = !!nValue;
                nForce = nActivate? nValue : nForce;
                nFlag += (1 << offset) * (nActivate);

                voice += abs(pValue) + abs(nValue);
            }

            auto chP = tag2CharPos[pFlag], chN = tag2CharNeg[nFlag];
            auto containsNull = (!chP || !chN);
            auto containsContradict = (pFlag == nFlag) && (pFlag != 0);

            if(containsNull || containsContradict) return result;
            thisSamplePattern += (pForce > nForce)? chP : chN;
            validLiteral++;
        }
        double value = (validLiteral / (double)(posPartLen / (double)2) * voice); // valid literal ratio multiply to voice.
        result[sampleIdx] = thisSamplePattern + "," + std::to_string(value);
    }
    return result;
}

/// @brief Deparse specified subset of trained tsetlin model clause(represented in integers) into GC content.
/// @param rawSignals Trained Tsetlin Machine model.
/// @param start Start index of continuous GC content literals in each clause.
/// @param end End index of continuous GC content literals in each clause.
/// @return Description of GC content tier pattern.
vector<string> Deparser::deparseGCSig2Seq(const vector<vector<int>> &rawSignals, const int start, const int end)
{
    std::unordered_map<int, string> gcMapPos, gcMapNeg;
    auto sampleNum = rawSignals.size();
    auto posPartLen = rawSignals[0].size()/2; // Represent tsetlin positive literal number.
    vector<string> result(sampleNum);

    gcMapPos.insert({0, "< " + std::to_string(discreteThreshold[0][0])});
    gcMapNeg.insert({0, "not < " + std::to_string(discreteThreshold[0][0])});
    string prevCut= std::to_string(discreteThreshold[0][0]);
    for(int i = 0; i < 2; i++)
    {
        string thisCut = std::to_string(discreteThreshold[0][i + 1]);
        gcMapPos.insert({1 << i, prevCut + " ~ " + thisCut});
        gcMapNeg.insert({1 << i, "Not in "+ prevCut + " ~ " + thisCut});
        prevCut = thisCut;
    }
    gcMapPos.insert({1<<(2), " > "  + prevCut});
    gcMapNeg.insert({1<<(2), "Not > "  + prevCut});
    
    for (auto sampleIdx = 0; sampleIdx < sampleNum; sampleIdx++)
    {
        string thisSamplePattern;
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
                pValue = rawSignals[sampleIdx][baseIdx + offset];
                pActivate = !!pValue;
                pForce =  pActivate ? pValue : pForce; // stay the same or be valid.
                pFlag += (1 << offset) * (pActivate);

                nValue= rawSignals[sampleIdx][baseIdx + offset + posPartLen];
                nActivate = !!nValue;
                nForce = nActivate? nValue : nForce;
                nFlag += (1 << offset) * (nActivate);

                voice += abs(pValue) + abs(nValue);
            }

            auto chP = gcMapPos[pFlag], chN = gcMapNeg[nFlag];
            auto containsNull = (chP.empty() || chN.empty());
            auto containsContradict = (pFlag == nFlag) && (pFlag != 0);

            if(containsNull || containsContradict) return result;
            thisSamplePattern += (pForce > nForce)? chP : chN;
            validLiteral++;
        }
        double value = (validLiteral / (double)(posPartLen / (double)2) * voice); // valid literal ratio multiply to voice.
        result[sampleIdx] = thisSamplePattern + "," + std::to_string(value);
    }
    return result;
}
