#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include "TsetlinMachine.h"
#include "pcg_random.hpp"

#define COMPLETED true
#define FAILED false
using std::vector;
using std::string;
vector<int> siRNA2SIG(string raw_string);

// TODO: Balance size of each class, add shuffle.
struct dataset
{
    //////////METADATA///////////////
    int             trainSize;
    int             testSize;
    int             responseSize;
    vector<double>     responseThreshold;
    //////////METADATA///////////////

    vector<vector<int>>     trainData;
    vector<vector<int>>     trainResponse;
    vector<vector<int>>     testData;
    vector<vector<int>>     testResponse;
    dataset(){}
};


void encodeHueskenSeqs(string path, 
                        vector<vector<int>> &result);

void encodeHueskenScores(string path, 
                          vector<vector<int>> &result);

template<typename Dtype>
vector<Dtype> readcsvline(string path)
{
    vector<Dtype> result;
    Dtype curr;
    string currStr;
    std::stringstream convert;
    std::ifstream file(path);
    while(std::getline(file,currStr))
    {
        convert.clear();
        convert<< currStr;
        convert>>curr;
        result.push_back(curr);
    }
    return result;
}
vector<int> getDiscreteResponse(vector<double> threshold, double raw);
dataset prepareData(vector<string> &seqs,vector<double> &responses, double trainRatio, int classes);

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