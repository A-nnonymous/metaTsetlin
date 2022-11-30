

#include <vector>
#include <string>
#include <string_view>
#include <functional>
#include "io.h"

using std::vector;
using std::string;

typedef std::function<vector<double>(const vector<string>&)> seqFeatureFunc;

struct pattern
{
    string sequence;
    double value;
    pattern operator=(pattern other)
    {
        this->sequence = other.sequence;
        this->value = other.value;
        return *this;
    }
};

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



template<typename... func>
struct seqWithFeatures
{
    const vector<string>        &rawSeq;                // N nucleotide sequences.
    vector<vector<double>>      additionFeatures;   // DxN additional features.
    seqWithFeatures(vector<string> &seq, func&... f):rawSeq(seq)
    {
        seqFeatureFunc funcs[]{f...};
        for(auto thisFunc: funcs)
        {
            additionFeatures.push_back(thisFunc(rawSeq));
        }
    }
};
vector<int> siRNA2SIG(const string &raw_string)noexcept;

vector<double> seqGC(const vector<string> &rawStrings)noexcept;
/// @brief Concatentate SiRNA signal and it's feature to train-ready 2-D vector
/// @tparam ...func packed parameter of feature functions
/// @param rawData Data stucture contain both sequences and additinal features.
/// @return / Train-ready vectors shaped in N * (seqSignalLen + featureLen)
template<typename... func>
vector<vector<int>> concateAndDiscrete(seqWithFeatures<func...> &rawData, int bitPerFeature)
{
    auto sampleNum = rawData.rawSeq.size();
    auto additionDimNum = rawData.additionFeatures.size();
    auto cutNum = bitPerFeature - 1;
    vector<vector<double>> featureThreshold(additionDimNum, vector<double>(cutNum,0));
    vector<vector<int>> result(sampleNum);

    for (auto addition = 0; addition < additionDimNum; addition++)
    {
        featureThreshold[addition] = getFairThreshold<double>(rawData.additionFeatures[addition], bitPerFeature);
        feature2StrMap[addition].insert({0, "< " + std::to_string(featureThreshold[addition][0])});
        string prevCut = std::to_string(featureThreshold[addition][0]);
        for(int i = 0; i < cutNum - 1; i++)
        {
            string thisCut = std::to_string(featureThreshold[addition][i + 1]);
            feature2StrMap[addition].insert({1 << i, prevCut + " ~ " + thisCut});
            prevCut = thisCut;
        }
        feature2StrMap[addition].insert({1<<(cutNum-1), " > "  + prevCut});
    }
    
    for(auto sample = 0; sample < sampleNum; sample++)
    {
        result[sample] = siRNA2SIG(rawData.rawSeq[sample]); // Raw nucleotide encoding.
        for (auto addition = 0; addition < additionDimNum; addition++)
        {
            auto thisFeatures = getDiscreteResponse(featureThreshold[addition],
                                                    rawData.additionFeatures[addition][sample]);
            result[sample].insert(  result[sample].end(),
                                    thisFeatures.begin(),
                                    thisFeatures.end());
        }
    }
    return result;
}



dataset prepareData(vector<string> &seqs,vector<double> &responses, double trainRatio, int classes);

pattern clause2NucPattern(vector<int> &signal);
pattern clause2FeaturePattern(vector<int> &signal,int funcN, int offset, int bitNum);

// Model Interpreting
void outputModelStat(TsetlinMachine::model  &machine,
                    double                  Precision,
                    vector<string>          tierTags,
                    string                  outputPath);

void outputModelPattern(TsetlinMachine::model   &machine,
                        double                  precision,
                        vector<string>          tierTags,
                        string                  outputPath);