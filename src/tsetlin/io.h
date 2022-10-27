#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include "TsetlinMachine.h"

#define COMPLETED true
#define FAILED false

std::vector<int> siRNA2SIG(std::string raw_string);

void parse_huesken_seqs(std::string path, 
                        std::vector<std::vector<int>> &result);

void parse_huesken_scores(std::string path, 
                          std::vector<std::vector<int>> &result);


bool write_csv_row(std::vector<float> data, std::ofstream *output);

template<typename dtype>
bool write_csv(dtype *data, int row, int column, std::string filepath)
{ 
    std::ofstream output;
    output.open(filepath + ".csv", std::ios::out);
    std::cout << "Output file stream opening success. " << std::endl;
    for(int j = 0; j < row; j++)
    {
        for (int i = 0; i < column - 1; i++)
        {
            output << data[j * column + i] << ",";
        }
        output << data[j*column + column-1]<<"\n";
    }
    output.close();
    return COMPLETED;
    
}

void
modelOutput (TsetlinMachine::model model,
             double precision,
             std::string outputpath);