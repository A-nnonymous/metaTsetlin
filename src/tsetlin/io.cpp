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
        output << data[j][column-1]<<"\n";
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
    std::cout << "Output file stream opening success. " << std::endl;
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
          result.insert (result.end (), { 0, 0, 0, 1 });
          break;
        case 'U':
          result.insert (result.end (), { 0, 0, 1, 0 });
          break;
        case 'C':
          result.insert (result.end (), { 0, 1, 0, 0 });
          break;
        case 'G':
          result.insert (result.end (), { 1, 0, 0, 0 });
          break;
        case 'a':
          result.insert (result.end (), { 0, 0, 0, 1 });
          break;
        case 't':
          result.insert (result.end (), { 0, 0, 1, 0 });
          break;
        case 'c':
          result.insert (result.end (), { 0, 1, 0, 0 });
          break;
        case 'g':
          result.insert (result.end (), { 1, 0, 0, 0 });
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

/// @brief Output human interpretable datas to downstream analysis, 
///        including most determined clauses, average knowledge(by inclusion), average knowledge(by weight)
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

    vector<vector<double>> posAvgWeight(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));   // Averaging weight of each clause.
    vector<vector<double>> negAvgWeight(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));
    vector<vector<double>> posAvgInc(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));   // Averaging weight of each clause.
    vector<vector<double>> negAvgInc(vector<vector<double>>(2*wordSize,
                                                            vector<double>(wordNum,0)));
    for (int tier = 0; tier < tierNum; tier++)
    {
        for (int clauseIdx = 0; clauseIdx < clausePerTier; clauseIdx++)
        {
            auto positive = decodeSeqs(machine.automatas[tier].positiveClauses[clauseIdx].literals,4);
            auto negative = decodeSeqs(machine.automatas[tier].negativeClauses[clauseIdx].literals,4);
            for (int nucType = 0; nucType < 2*wordSize ;nucType++)  // Statistic affairs
            {
                for (int wordIdx = 0; wordIdx < wordNum; wordIdx++)
                {
                    posAvgWeight[nucType][wordIdx] += positive[nucType][wordIdx] / (double)clausePerTier;
                    negAvgWeight[nucType][wordIdx] += negative[nucType][wordIdx] / (double)clausePerTier;
                    posAvgInc[nucType][wordIdx] += (positive[nucType][wordIdx]>=0? 0:1) / (double)clausePerTier;
                    negAvgInc[nucType][wordIdx] += (negative[nucType][wordIdx]>=0? 0:1) / (double)clausePerTier;
                }
            }
        }
        string prefix = outputPath + "/p" + std::to_string(Precision) + "/" + tierTags[tier] +"/";
        std::filesystem::create_directories(prefix);
        write_csv(posAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgWeight");
        write_csv(negAvgWeight,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgWeight");
        write_csv(posAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "posAvgInc");
        write_csv(negAvgInc,2*wordSize,wordNum,false,vector<string>(),prefix + "negAvgInc");
    }

}

/*
void
modelOutput (TsetlinMachine::model model,
             double precision,
             std::string outputpath)
{
  
    int clausePerOutput = model.modelArgs.clausePerOutput;
    int literalNum = model.modelArgs.inputSize;

    vector<vector<int>> rare,mediumRare,mediumWell,wellDone;
    vector<vector<int>> rareNegative,mediumRareNegative,mediumWellNegative,wellDoneNegative;
    rare.resize(clausePerOutput, vector<int>(literalNum * 2,0));          // Clauses that approve of silence score less than 0.5;
    mediumRare.resize(clausePerOutput, vector<int>(literalNum * 2,0));    // Clauses that approve of silence score less than 0.7 but larger or equal to 0.5;
    mediumWell.resize(clausePerOutput, vector<int>(literalNum * 2,0));    // Clauses that approve of silence score less than 0.9 but larger or equal to 0.7;
    wellDone.resize(clausePerOutput, vector<int>(literalNum * 2,0));      // Clauses that approve of silence score larger than 0.9;

    rareNegative.resize(clausePerOutput, vector<int>(literalNum * 2,0));          // Clauses that disapprove of silence score less than 0.5;
    mediumRareNegative.resize(clausePerOutput, vector<int>(literalNum * 2,0));    // Clauses that disapprove of silence score less than 0.7 but larger or equal to 0.5;
    mediumWellNegative.resize(clausePerOutput, vector<int>(literalNum * 2,0));    // Clauses that disapprove of silence score less than 0.9 but larger or equal to 0.7;
    wellDoneNegative.resize(clausePerOutput, vector<int>(literalNum * 2,0));      // Clauses that disapprove of silence score larger than 0.9;
    
    for (int clauseIdx = 0; clauseIdx < clausePerOutput; clauseIdx++)
    {
        for (int literalIdx = 0; literalIdx < literalNum; literalIdx++)
        {
            rare[clauseIdx][literalIdx] = model.automatas[3].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
            rare[clauseIdx][literalIdx + literalNum] = model.automatas[3].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
            mediumRare[clauseIdx][literalIdx] = model.automatas[2].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumRare[clauseIdx][literalIdx + literalNum] = model.automatas[2].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
            mediumWell[clauseIdx][literalIdx] = model.automatas[1].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumWell[clauseIdx][literalIdx + literalNum] = model.automatas[1].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
            wellDone[clauseIdx][literalIdx] = model.automatas[0].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
            wellDone[clauseIdx][literalIdx + literalNum] = model.automatas[0].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
            rareNegative[clauseIdx][literalIdx] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            rareNegative[clauseIdx][literalIdx + literalNum] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumRareNegative[clauseIdx][literalIdx] = model.automatas[2].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumRareNegative[clauseIdx][literalIdx + literalNum] = model.automatas[2].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumWellNegative[clauseIdx][literalIdx] = model.automatas[1].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            mediumWellNegative[clauseIdx][literalIdx + literalNum] = model.automatas[1].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            wellDoneNegative[clauseIdx][literalIdx] = model.automatas[0].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
            wellDoneNegative[clauseIdx][literalIdx + literalNum] = model.automatas[0].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
        }
    }
    write_csv (rare, clausePerOutput, literalNum,
                    outputpath + "./rare_" + std::to_string(precision));
    write_csv (rareNegative, clausePerOutput, literalNum,
                    outputpath + "./rare_negative_" + std::to_string(precision));
    write_csv (mediumRare, clausePerOutput, literalNum,
                    outputpath + "./mediumRare_" + std::to_string(precision));
    write_csv (mediumRareNegative, clausePerOutput, literalNum,
                    outputpath + "./mediumRare_negative_" + std::to_string(precision));
    write_csv (mediumWell, clausePerOutput, literalNum,
                    outputpath + "./mediumWell_" + std::to_string(precision));
    write_csv (mediumWellNegative, clausePerOutput, literalNum,
                    outputpath + "./mediumWell_negative_" + std::to_string(precision));
    write_csv (wellDone, clausePerOutput, literalNum,
                    outputpath + "./wellDone_" + std::to_string(precision));
    write_csv (wellDoneNegative, clausePerOutput, literalNum,
                    outputpath + "./wellDone_negative_" + std::to_string(precision));
}
*/