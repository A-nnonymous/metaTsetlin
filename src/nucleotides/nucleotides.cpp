#include "nucleotides.h"
using std::string_view;

/// @brief Compute all sequences' GC content.
/// @param rawStrings Vector of all nucleotide sequences.
/// @return Vector of GC contents corresponding to all sequences.
vector<double> seqGC(const vector<string> &rawStrings)noexcept
{
    string_view sv;
    auto sampleNum = rawStrings.size();
    vector<double> result(sampleNum, 0);
    for(auto idx = 0; idx < sampleNum; idx++)
    {
        sv = string_view(rawStrings[idx]);
        result[idx] = ( std::count(sv.begin(), sv.end(),'G') +
                        std::count(sv.begin(), sv.end(),'C') ) / (double)sv.size();
    }
    result.shrink_to_fit();
    return result;
}


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


/// @brief Transform tsetlin clauses into siRNA pattern clauses, appended with pattern's value.
/// @param signal Trained tsetlin clause contain a vector of int.
/// @return A pattern string with value.
pattern
clause2NucPattern(vector<int> &signal, int nucLenth)
{
    string sequence;
    pattern result;

    if(signal.size()%4 != 0)
    {
        std::cout<< "Invalid signal length for transformation"<<std::endl;
        return result;
    }
    int validLiteral = 0;
    long voice =0;
    for (int p = 0; p < 4 * nucLenth; p+=4)
    {
        int pFlag=0, nFlag=0;
        int pforce=0, nforce=0;     // represent literal force if bipolar-literal not contradict and both exist.
        bool pactivate, nactivate;
        for(int i=0; i< 4; i++)
        {
            pactivate = signal[p + i] >= 0;
            pforce = pactivate? signal[p + i] : pforce;
            pFlag += (1<<i) * (pactivate);
            nactivate = signal[p + signal.size()/2 + i] >= 0;
            nforce = nactivate? signal[p + signal.size()/2 +i] : nforce;
            nFlag += (1<<i) * (nactivate);
            voice += abs(signal[p + i]) + abs(signal[p + signal.size()/2 + i]);
        }
        auto chP = tag2CharPos[pFlag], chN = tag2CharNeg[nFlag];
        auto containsNull = (!chP || ! chN);
        auto containsContradict = (pFlag == nFlag) && (pFlag != 0);
        if(containsNull || containsContradict) return result;
        sequence += (pforce > nforce)? chP : chN;
        validLiteral++;
    }
    double value = (validLiteral / (double)(signal.size()/4)) * voice; // valid literal ratio multiply to voice.
    result.value = value;
    result.sequence = sequence;
    return result;
}
pattern clause2FeaturePattern(vector<int> &signal,int funcN, int offset, int bitNum)
{
    string sequence;
    pattern result;
    bool pactivate, nactivate;
    int pforce = 0, nforce = 0;
    int pflag = 0, nflag = 0;
    long voice = 0;
    for (int i = offset; i < offset + bitNum; i++)
    {
        pactivate = signal[i] >= 0;
        pforce = pactivate? signal[i] :pforce;
        pflag += (1 << (i - offset)) * pactivate;
        nactivate = signal[i + signal.size()/2] >= 0;
        nforce = nactivate? signal[i +signal.size()] :nforce;
        nflag += (1 << (i - offset)) * nactivate;
        voice += abs(signal[i]) + abs(signal[i + signal.size()/2]);
    }
    auto strP = feature2StrMap[funcN][pflag], strN = feature2StrMap[funcN][nflag];
    auto containsNull = (strP.empty() || strN.empty());
    auto containContradict = (pflag == nflag) && (pflag != 0);
    if(containsNull || containContradict) return result;
    sequence += (pforce > nforce)? strP : strN;
    result.value = (pforce > nforce) ? (double)pforce : (double)nforce;
    result.sequence = sequence;
    return result;
}

/// @brief Prepare dataset for training and testing, using a datafile sorted by response.
/// @param path FilePath of the whole raw sorted dataset
/// @param trainRatio Size ratio of trainset/testset.
/// @param classes Number of classes.
/// @return A formatted dataset containing 'metadata'.
dataset prepareData(vector<string> &seqs,vector<double> &responses, double trainRatio, int classes)
{
    ////////// arg check ////////////
    bool isRightLength = seqs.size() == responses.size();
    bool isPossibleDivision = classes <= seqs.size();
    if(!isRightLength)std::cout<<"Data length corrupted."<<std::endl;
    if(!isPossibleDivision)std::cout<<"Can't divide data into "<<classes<<" parts."<<std::endl;
    if(!isRightLength || !isPossibleDivision)return dataset();
    ////////// arg check ////////////
    
    int datasize = seqs.size();
    seqFeatureFunc gcFunc(seqGC);
    seqWithFeatures<seqFeatureFunc> data(seqs,gcFunc);
    auto all = concateAndDiscrete(data, 4);
    for(int i = 0; i < 4; i++)
    {
        std::cout<< feature2StrMap[0][(!!i)? 1<<(i-1) : 0] << std::endl;
    }

    ////////// Response threshold picking////////////
    auto responseThreshold = getFairThreshold<double> (responses, classes);
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
            trainData.push_back(all[i]);
            trainResponse.push_back(thisResponse);
        }
        else[[unlikely]]            // Picked to test set.
        {
            testData.push_back(all[i]);
            testResponse.push_back(thisResponse);
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

/// @brief Decode clause from trained tsetlin machine, and form a nucleotide weight matrix in size of 8*21
/// @param original Original trained result extract from tsetlin machine.
/// @param wordSize Costs of bit in each position of nucleotide, for this implement is 4.
/// @return A 2d vector, each row represent a type of nucleotide(or its negation).
vector<vector<int>> 
decodeSeqs( vector<int> &original, int wordSize)
{
    int seqLen = original.size() / (2 * wordSize);
    vector<vector<int>> result(wordSize* 2 , vector<int>(seqLen, 0));
    for (int i = 0; i < original.size()/2; i++)
    {
        int thisWord = i % wordSize;
        int thisPos = i/wordSize;
        result[thisWord][thisPos] = original[i];                                //positive literal.
        result[thisWord + wordSize][thisPos] = original[i + original.size()/2]; //negative literal(asserting word size is equal to variation of words).
    }
    
    return result;

}

/// @brief Output human interpretable datas for downstream analysis, 
///        including average clause knowledge(by inclusion possibility or by literal weight)
/// @param machine Trained tsetlin machine.
/// @param Precision Precision of this trained model, using this to name the directory.
/// @param tierTags Tag of each tier, using this to name the sub-directory.
/// @param outputPath The workspace of this output process.
void 
outputModelStat(    TsetlinMachine::model   &machine,
                    double                  Precision,
                    vector<string>          tierTags,
                    string                  outputPath) [[_GLIBCXX20_DEPRECATED]]
{
    if(!outputPath.ends_with("/"))outputPath+="/";
    std::filesystem::create_directories(outputPath);
    ////////////////Fixed variable due to laziness//////////////
    int wordSize = 4;
    ////////////////Fixed variable due to laziness//////////////
    int clausePerTier = machine.modelArgs.clausePerOutput; // only represent clause number of a single polarity.
    int literalNum = machine.modelArgs.inputSize; // same as above.
    int wordNum = literalNum / wordSize;
    int tierNum = machine.modelArgs.outputSize;

    for (int tier = 0; tier < tierNum; tier++)
    {
        vector<vector<double>> posAvgWeight(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));   // Averaging weight of each clause.
        vector<vector<double>> negAvgWeight(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));
        vector<vector<double>> posAvgInc(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));   // Averaging weight of each clause.
        vector<vector<double>> negAvgInc(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));
        for (int clauseIdx = 0; clauseIdx < clausePerTier; clauseIdx++)
        {
            vector<vector<int>> positive = decodeSeqs(machine.automatas[tier].positiveClauses[clauseIdx].literals,4);
            vector<vector<int>> negative = decodeSeqs(machine.automatas[tier].negativeClauses[clauseIdx].literals,4);
            for (int nucType = 0; nucType < 2*wordSize ;nucType++)  // Statistic affairs
            {
                for (int wordIdx = 0; wordIdx < wordNum; wordIdx++)
                {
                    posAvgWeight[nucType][wordIdx] += (positive[nucType][wordIdx] / (double)clausePerTier);
                    negAvgWeight[nucType][wordIdx] += (negative[nucType][wordIdx] / (double)clausePerTier);
                    posAvgInc[nucType][wordIdx] += ((positive[nucType][wordIdx]>=0? 1:0) / (double)clausePerTier);
                    negAvgInc[nucType][wordIdx] += ((negative[nucType][wordIdx]>=0? 1:0) / (double)clausePerTier);
                }
            }
        }
        string prefix = outputPath + "/prec" + std::to_string(Precision) + "/" + tierTags[tier] +"/";
        std::filesystem::create_directories(prefix);
        write_csv<double>(posAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgWeight");
        write_csv<double>(negAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgWeight");
        write_csv<double>(posAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgInc");
        write_csv<double>(negAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgInc");
    }
}

/// @brief Decode and output all legal clauses(this interpretion use only in Huesken dataset).
/// @param machine Target tsetlin machine model.
/// @param outputPath Path that store pattern data csv files.
void outputModelPattern(TsetlinMachine::model   &machine,
                        double                  precision,
                        vector<string>          tierTags,
                        string                  outputPath)[[_GLIBCXX20_DEPRECATED]]
{
    if(!outputPath.ends_with("/"))outputPath+="/";
    std::filesystem::create_directories(outputPath);
    ////////////////Fixed variable due to laziness//////////////
    int wordSize = 4;
    ////////////////Fixed variable due to laziness//////////////
    int clausePerTier = machine.modelArgs.clausePerOutput; // only represent clause number of a single polarity.
    int literalNum = machine.modelArgs.inputSize; // same as above.
    int wordNum = literalNum / wordSize;
    int tierNum = machine.modelArgs.outputSize;

    for (int tier = 0; tier < tierNum; tier++)
    {
        vector<vector<string>> positiveClauses; //Initialize empty strings for clauses.
        vector<vector<string>> negativeClauses;
        vector<vector<string>> positiveClausesFeature;
        vector<vector<string>> negativeClausesFeature;
        
        for (int clauseIdx = 0; clauseIdx < clausePerTier; clauseIdx++)
        {
            auto posClause = machine.automatas[tier].positiveClauses[clauseIdx].literals;
            auto negClause = machine.automatas[tier].negativeClauses[clauseIdx].literals;
            vector<string> positive(2);
            vector<string> positiveFeature(2);
            vector<string> negative(2);
            vector<string> negativeFeature(2);
            pattern positivePattern = clause2NucPattern(posClause,21);
            pattern positiveFeaturePattern= clause2FeaturePattern(posClause,0,84,4);
            pattern negativePattern = clause2NucPattern(negClause,21);
            pattern negativeFeaturePattern = clause2FeaturePattern(negClause,0,84,4);
            if(!positivePattern.sequence.empty())
            {
                positive[0] = positivePattern.sequence;
                positive[1] = std::to_string(positivePattern.value);
                positiveClauses.push_back(positive);
            }
            if(!positiveFeaturePattern.sequence.empty())
            {
                positiveFeature[0] = positiveFeaturePattern.sequence;
                positiveFeature[1] = std::to_string(positiveFeaturePattern.value);
                positiveClausesFeature.push_back(positiveFeature);
            }
            if(!negativePattern.sequence.empty())
            {
                negative[0] = negativePattern.sequence;
                negative[1] = std::to_string(negativePattern.value);
                negativeClauses.push_back(negative);
            }
            if(!negativeFeaturePattern.sequence.empty())
            {
                negativeFeature[0] = negativeFeaturePattern.sequence;
                negativeFeature[1] = std::to_string(negativeFeaturePattern.value);
                negativeClausesFeature.push_back(negativeFeature);
            }
        }
        string prefix = outputPath + "/prec" + std::to_string(precision) + "/" + tierTags[tier] +"/";
        std::filesystem::create_directories(prefix);
        write_csv<string>(positiveClauses,positiveClauses.size(),2,false,vector<string>(),prefix + "posPattern");
        write_csv<string>(negativeClauses,negativeClauses.size(),2,false,vector<string>(),prefix + "negPattern");
        write_csv<string>(positiveClausesFeature,positiveClausesFeature.size(),2,false,vector<string>(),prefix + "negPatternFeature");
        write_csv<string>(negativeClausesFeature,negativeClausesFeature.size(),2,false,vector<string>(),prefix + "negPatternFeature");
    }
}