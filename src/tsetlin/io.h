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



void
modelOutput (TsetlinMachine::model model,
             double precision,
             std::string outputpath);