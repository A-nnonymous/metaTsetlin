#include "io.h"
using std::vector;
using std::string;


bool
write_csv_row (vector<float> data, std::ofstream *output)
{
    int i;
    for (i = 0; i < data.size () - 1; i++)
    {
        (*output) << data[i] << ",";
    }
    (*output) << data[i] << "\n";
    return COMPLETED;
}

bool write_csv(vector<vector<int>> &data, int row, int column, std::string filepath)[[deprecated]]
{ 
    std::ofstream output;
    output.open(filepath + ".csv", std::ios::out);
    std::cout << "Output file stream opening success. " << std::endl;
    for(int j = 0; j < row; j++)
    {
        for (int i = 0; i < column - 1; i++)
        {
            output << data[j][i] << ",";
        }
        if(j != row-1)
        {
            output << data[j][column-1]<<"\n";
        }
        else
        {
            output<< data[j][column-1];
        }
    }
    output.close();
    return COMPLETED;
    
}



vector<int>
siRNA2SIG (std::string raw_string)
{
  vector<int> result;
  size_t current_idx = 0;
  for (char c : raw_string)
    {
      switch (c)
        {
        case 'A':
            result.insert (result.end (), { 1, 0, 0, 0 });
            break;
        case 'U':
            result.insert (result.end (), { 0, 1, 0, 0 });
            break;
        case 'C':
            result.insert (result.end (), { 0, 0, 1, 0 });
            break;
        case 'G':
            result.insert (result.end (), { 0, 0, 0, 1 });
            break;
        case 'a':
            result.insert (result.end (), { 1, 0, 0, 0 });
            break;
        case 't':
            result.insert (result.end (), { 0, 1, 0, 0 });
            break;
        case 'c':
            result.insert (result.end (), { 0, 0, 1, 0 });
            break;
        case 'g':
            result.insert (result.end (), { 0, 0, 0, 1 });
            break;
        default: // Not expected other characters.
            break;
        }
    }
  return result;
}

/// @brief Transform tsetlin clauses into siRNA pattern clauses, appended with pattern's value.
/// @param signal Trained tsetlin clause contain a vector of int.
/// @return A pattern string with value.
pattern
clause2Pattern(vector<int> signal)
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
    for (int p = 0; p < signal.size()/2; p+=4)
    {
        int pFlag=0, nFlag=0, flag;
        bool isP;
        for(int i=0; i< 4; i++)
        {
            pFlag += (1<<i) * (signal[p + i] >= 0);
            nFlag += (1<<i) * (signal[p + signal.size()/2 + i] >= 0);
            voice += abs(signal[p + i]) + abs(signal[p + signal.size()/2 + i]);
        }
        if((pFlag!=0) && (nFlag!=0)) return result;
        isP = (pFlag == 0)? false:true;
        flag = (isP)? pFlag: nFlag;
        switch(flag)
        {
            case 0:[[likely]]   // sparse pattern, maybe.
                sequence += "_";  // for those literal that deactivated.
                break;
            case 1:
                if(isP)
                {
                    sequence += "A";
                }
                else
                {
                    sequence += "W"; // 'W' represent "!A", same a below.
                }
                break;
            case 2:
                if(isP)
                {
                    sequence += "U";
                }
                else
                {
                    sequence += "X"; 
                }
                break;
            case 4:
                if(isP)
                {
                    sequence += "C";
                }
                else
                {
                    sequence += "Y";
                }
                break;
            case 8:
                if(isP)
                {
                    sequence += "G";
                }
                else
                {
                    sequence += "Z"; 
                }
                break;
            default:
                return pattern();// Invalid clause because contain impossible literal.
        }
        if(flag != 0)validLiteral++; // valid literal counter.
    }
    double value = (validLiteral / (double)(signal.size()/4)) * voice; // valid literal ratio multiply to voice.
    result.value = value;
    result.sequence = sequence;
    return result;
}

void
encodeHueskenSeqs (std::string path, vector<vector<int> > &result)
{
    std::ifstream seqfile (path);
    std::string thisline;
    int row = 0;
    while (std::getline (seqfile, thisline))
    {
        result[row++] = siRNA2SIG (thisline);
    }
}

void
encodeHueskenScores (std::string path, vector<vector<int> > &result)
{
    std::ifstream score_file (path);
    std::string score_string;
    float this_score;
    int idx = 0;
    while (std::getline (score_file, score_string))
    {
        this_score = ::atof (score_string.c_str ());
        if (this_score < 0.5)[[unlikely]]
            result[idx++] = { 0, 0, 0, 1 };
        if (this_score >= 0.5 && this_score < 0.7)[[likely]]
            result[idx++] = { 0, 0, 1, 0 };
        if (this_score >= 0.7 && this_score < 0.9)[[likely]]
            result[idx++] = { 0, 1, 0, 0 };
        if (this_score >= 0.9)[[unlikely]]
            result[idx++] = { 1, 0, 0, 0 };
    }
}

/// @brief Convert continuous data to discrete through grey-scale like threshold.
/// @param threshold Vector of incremental continuous threshold.
/// @param raw Raw continuous data.
/// @return A vector of discrete data.
vector<int> getDiscreteResponse(vector<double> threshold, double raw)
{
    vector<int> result(threshold.size()+1, 0);
    for (int i = 0; i < threshold.size(); i++)
    {
        double thiscut = threshold[i];
        if(raw <= thiscut)
        {
            result[i] = 1;
            return result;
        }
    }
    result[result.size()-1] = 1; // larger than final cut.
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

    ////////// Threshold picking////////////
    vector<double> threshold(classes - 1,0);   // N classes, N-1 cut.
    int stdLen = datasize / classes;
    int remain = datasize % classes;

    int end = 0;
    for(int cut = 0; cut < classes - 1; cut++)
    {
        end += (remain>0)? (stdLen + !!(remain--)) :stdLen;
        threshold[cut] = responses[end];
    }
    ////////// Threshold picking////////////

    ///////////////////// random chosing data//////////////////////
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
        thisResponse = getDiscreteResponse(threshold, responses[i]);
        if(d(_rng)==1)[[likely]]    // Picked to training set.
        {
            trainData.push_back(siRNA2SIG(seqs[i]));
            trainResponse.push_back(thisResponse);
        }
        else[[unlikely]]            // Picked to test set.
        {
            testData.push_back(siRNA2SIG(seqs[i]));
            testResponse.push_back(thisResponse);
        }
    }
    ///////////////////// random chosing data//////////////////////
    
    result.responseSize = classes;
    result.responseThreshold = threshold;
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
    for (int i = 0; i < threshold.size(); i++)
    {
        std::cout<< threshold[i]<<"\t";
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
                    string                  outputPath)
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
                        string                  outputPath)
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
        //vector<vector<string>> positiveClauses(clausePerTier, vector<string>(2, string())); //Initialize empty strings for clauses.
        //vector<vector<string>> negativeClauses(clausePerTier, vector<string>(2, string()));
        vector<vector<string>> positiveClauses; //Initialize empty strings for clauses.
        vector<vector<string>> negativeClauses;
        
        for (int clauseIdx = 0; clauseIdx < clausePerTier; clauseIdx++)
        {
            vector<string> positive(2);
            vector<string> negative(2);
            pattern positivePattern = clause2Pattern(machine.automatas[tier].positiveClauses[clauseIdx].literals);
            pattern negativePattern = clause2Pattern(machine.automatas[tier].negativeClauses[clauseIdx].literals);
            if(!positivePattern.sequence.empty())
            {
                //positiveClauses[clauseIdx][0] = positivePattern.sequence;
                //positiveClauses[clauseIdx][1] = std::to_string(positivePattern.value);
                positive[0] = positivePattern.sequence;
                positive[1] = std::to_string(positivePattern.value);
                positiveClauses.push_back(positive);
            }
            if(!negativePattern.sequence.empty())
            {
                //negativeClauses[clauseIdx][0] = negativePattern.sequence;
                //negativeClauses[clauseIdx][1] = std::to_string(negativePattern.value);
                negative[0] = negativePattern.sequence;
                negative[1] = std::to_string(negativePattern.value);
                negativeClauses.push_back(negative);
            }
        }
        string prefix = outputPath + "/prec" + std::to_string(precision) + "/" + tierTags[tier] +"/";
        std::filesystem::create_directories(prefix);
        write_csv<string>(positiveClauses,positiveClauses.size(),2,false,vector<string>(),prefix + "posPattern");
        write_csv<string>(negativeClauses,negativeClauses.size(),2,false,vector<string>(),prefix + "negPattern");
    }
}

/// @brief Save Tsetlin machine model in binary format.
/// @param machine Target model.
/// @param outputPath Path of output file.
void saveModel( TsetlinMachine::model   &machine,
                string                  outputPath)
{
    write_binary<TsetlinMachine::model>(&machine,1,outputPath);
}

/// @brief Load Tsetlin machine model from binary model file.
/// @param modelPath Path of model file.
/// @return A structured model of Tsetlin machine.
TsetlinMachine::model loadModel(string modelPath)
{
    TsetlinMachine::model result;
    read_binary<TsetlinMachine::model>(modelPath, &result);
    return result;
}

