#include "io.h"

bool
write_csv_row (std::vector<float> data, std::ofstream *output)
{
    int i;
    for (i = 0; i < data.size () - 1; i++)
    {
        (*output) << data[i] << ",";
    }
    (*output) << data[i] << "\n";
    return COMPLETED;
}

std::vector<int>
siRNA2SIG (std::string raw_string)
{
  std::vector<int> result;
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
parse_huesken_seqs (std::string path, std::vector<std::vector<int> > &result)
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
parse_huesken_scores (std::string path, std::vector<std::vector<int> > &result)
{
    std::ifstream score_file (path);
    std::string score_string;
    float this_score;
    int idx = 0;
    while (std::getline (score_file, score_string))
    {
        this_score = ::atof (score_string.c_str ());
        if (this_score < 0.3)
            result[idx++] = { 0, 0, 0, 1 };
        if (this_score >= 0.3 && this_score < 0.5)
            result[idx++] = { 0, 0, 1, 0 };
        if (this_score >= 0.5 && this_score < 0.7)
            result[idx++] = { 0, 1, 0, 0 };
        if (this_score >= 0.7)
            result[idx++] = { 1, 0, 0, 0 };
    }
}

void
modelOutput (TsetlinMachine::model model,
             double precision,
             std::string outputpath)
{

}