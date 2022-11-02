#include "io.h"
using std::vector;
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
bool write_csv(vector<vector<int>>data, int row, int column, std::string filepath)
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
parse_huesken_seqs (std::string path, vector<vector<int> > &result)
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
parse_huesken_scores (std::string path, vector<vector<int> > &result)
{
    std::ifstream score_file (path);
    std::string score_string;
    float this_score;
    int idx = 0;
    while (std::getline (score_file, score_string))
    {
        this_score = ::atof (score_string.c_str ());
        if (this_score < 0.4)
            result[idx++] = { 0, 0, 0, 1 };
        if (this_score >= 0.4 && this_score < 0.6)
            result[idx++] = { 0, 0, 1, 0 };
        if (this_score >= 0.6 && this_score < 0.8)
            result[idx++] = { 0, 1, 0, 0 };
        if (this_score >= 0.8)
            result[idx++] = { 1, 0, 0, 0 };
    }
}

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
          mediumRare[clauseIdx][literalIdx] = model.automatas[3].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumRare[clauseIdx][literalIdx + literalNum] = model.automatas[3].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
          mediumWell[clauseIdx][literalIdx] = model.automatas[3].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumWell[clauseIdx][literalIdx + literalNum] = model.automatas[3].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
          wellDone[clauseIdx][literalIdx] = model.automatas[3].positiveClauses[clauseIdx].positiveLiterals[literalIdx];
          wellDone[clauseIdx][literalIdx + literalNum] = model.automatas[3].positiveClauses[clauseIdx].negativeLiterals[literalIdx];
          
          rareNegative[clauseIdx][literalIdx] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          rareNegative[clauseIdx][literalIdx + literalNum] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumRareNegative[clauseIdx][literalIdx] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumRareNegative[clauseIdx][literalIdx + literalNum] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumWellNegative[clauseIdx][literalIdx] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          mediumWellNegative[clauseIdx][literalIdx + literalNum] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          wellDoneNegative[clauseIdx][literalIdx] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
          wellDoneNegative[clauseIdx][literalIdx + literalNum] = model.automatas[3].negativeClauses[clauseIdx].positiveLiterals[literalIdx];
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