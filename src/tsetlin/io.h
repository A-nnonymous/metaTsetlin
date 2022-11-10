#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <filesystem>
#include "TsetlinMachine.h"

#define COMPLETED true
#define FAILED false
using std::vector;
using std::string;
vector<int> siRNA2SIG(string raw_string);

void encodeHueskenSeqs(string path, 
                        vector<vector<int>> &result);

void encodeHueskenScores(string path, 
                          vector<vector<int>> &result);


bool write_csv_row(vector<float> data, std::ofstream *output);
bool write_csv(vector<vector<int>> &data, int row, int column, std::string filepath);
bool write_csv( vector<vector<double>> &data,
                int row, int column,
                bool isHeaderExist,vector<string> headers,
                string filepath);


/*
void
modelOutput (TsetlinMachine::model model,
             double precision,
             std::string outputpath);
*/
void 
modelOutputStat(    TsetlinMachine::model   &machine,
                    double                  Precision,
                    vector<string>          tierTags,
                    string                  outputPath);