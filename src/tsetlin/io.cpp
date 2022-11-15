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
bool write_csv(vector<vector<int>> &data, int row, int column, std::string filepath)
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
bool write_csv( vector<vector<double>> &data,
                int row, int column,
                bool isHeaderExist,vector<string> headers,
                string filepath)
{ 
    std::ofstream output;
    output.open(filepath + ".csv", std::ios::out);
    //std::cout << "Output file stream opening success. " << std::endl;
    if(isHeaderExist)
    {
        for(int j = 0; j <= row; j++)
        {
            if(j==0)[[unlikely]]// headers
            {
                for (int i = 0; i < column - 1; i++)
                {
                    output<< headers[i]<<",";
                }
                output<< headers[column-1] <<"\n";
            }
            else[[likely]]
            {
                for (int i = 0; i < column - 1; i++)
                {
                    output << data[j-1][i] << ",";
                }
                output << data[j-1][column-1]<<"\n";
            }
        }
    }
    else
    {
        for(int j = 0; j < row; j++)
        {
            for (int i = 0; i < column - 1; i++)
            {
                output << data[j][i] << ",";
            }
            output << data[j][column-1]<<"\n";
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
modelOutputStat(    TsetlinMachine::model   &machine,
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
        write_csv(posAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgWeight");
        write_csv(negAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgWeight");
        write_csv(posAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgInc");
        write_csv(negAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgInc");
    }

}

/// @brief Output weight vectors that can be reused by tsetlin machine
/// @param model Target data structure that meaned to export
/// @param precision Performance of this model
/// @param outputpath Directory of output files.
void
TMmodelExport(TsetlinMachine::model model,
             double precision,
             std::string outputpath)
{

}

/// @brief Build and return a tsetlin machine according to existing model file.
/// @param modelPath 
/// @return 
TsetlinMachine
TMbuildFromModel(string modelPath)
{
    
}