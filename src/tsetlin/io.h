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


// data is merely a vector of int concatenated by sequence and it`s additional features.

struct dataset
{
    //////////METADATA///////////////
    int                     trainSize;
    int                     testSize;
    int                     responseSize;
    vector<double>          responseThreshold;
    vector<string>          tierTags;
    //////////METADATA///////////////

    vector<vector<int>>     trainData;
    vector<vector<int>>     trainResponse;
    vector<vector<int>>     testData;
    vector<vector<int>>     testResponse;
    dataset(){}
};


template<typename T>
vector<T> getFairThreshold(const vector<T> &original, const int pieces)
{
    vector<T> result(pieces-1,0);
    vector<T> cutReady;
    if(pieces <= 1)     // Invalid orphand
    {
        std::cout<<"invalid piece number"<<std::endl;
        return result;
    }
    if(!std::is_sorted(original.begin(),original.end())) // Unsorted original.
    {
        cutReady = original;
        std::sort(cutReady.begin(),cutReady.end());
    }
    else
    {
        cutReady = original;
    }
    
    size_t originLen = original.size();
    int ration = originLen / pieces;
    int remain = originLen % pieces;

    int end = 0;
    for(int cut = 0; cut < pieces- 1; cut++)
    {
        end += (remain>0)? (ration + !!(remain--)) : ration;
        result[cut] = cutReady[end];
    }
    return result;
}

vector<int> getDiscreteResponse(vector<double> threshold, double raw);


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


bool write_csv_row(vector<float> data, std::ofstream *output);
bool write_csv(vector<vector<int>> &data, int row, int column, std::string filepath);
bool write_csv( vector<vector<double>> &data,
                int row, int column,
                bool isHeaderExist,vector<string> headers,
                string filepath);

// TODO: Output model in binary and load in binary, also implement a transform function to csv.
//       output all clause 
vector<string> threshold2Tags(vector<double> thresholds, bool isAscent);
// Model IO(deprecated)
void saveModel( TsetlinMachine::model   &mahchine,
                string                  outputPath);

TsetlinMachine::model loadModel(string modelPath);

template<typename T>
bool write_csv( vector<vector<T>> &data,
                int row, int column,
                bool isHeaderExist,vector<string> headers,
                string filepath)
{ 
    std::ofstream output;
    output.open(filepath + ".csv", std::ios::out);
    //std::cout << "Output file stream opening success. " << std::endl;
    if(isHeaderExist) // Exist user-defined header.
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
                
                if(j!=row)[[likely]]
                {
                    output << data[j-1][column-1]<<"\n";
                }
                else[[unlikely]] // last row of data, without endline.
                {
                    output << data[j-1][column-1];
                }
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

            if(j!=(row-1))[[likely]]
            {
                output << data[j][column-1]<<"\n";
            }
            else[[unlikely]] // last row of data, without endline.
            {
                output << data[j][column-1];
            }
        }
    }
    output.close();
    return COMPLETED;
}

/// @brief Write specified data structure(or structure array) into binary file
/// @tparam dtype Data type of single instance of data.
/// @param data Pointer to target data.
/// @param length If output array of structure, this represent instance number in this array.
/// @param filepath Output file path.
template<typename dtype>
void write_binary(dtype *data, int length, std::string filepath)
{
    std::cout.precision(2);
    std::stringstream sstream;
    std::ofstream output;
    size_t byteNum = length * sizeof(dtype);
    std::cout << "=======================" << "\n";
    output.open(filepath, std::ios::out |std::ios::binary);
    std::cout << "Output file stream for " << filepath << " is successfully opened. " << "\n";
    std::cout << "Estimated storage usage: " << std::fixed << byteNum / (double)(1024 * 1024) << "MB" << "\n";

    std::cout << "Start to output......" << "\n";
    output.write((char*)data, byteNum);
    output.close();
    std::cout << "Output for " << filepath << " is completed. " << "\n";
    std::cout << "\n";
}

/// @brief Read binary file into specified data structure(or structure array)
/// @tparam dtype Data type of single instance of data.
/// @param filepath Output file path.
/// @param outside_array_pointer Pointer provided by caller of this function, must allocated space already.
template<typename dtype>
void read_binary(std::string filepath, dtype* outside_array_pointer)
{
    std::ifstream fastseek(filepath, std::ios::binary | std::ios::ate);
    size_t byteNum = fastseek.tellg();
    fastseek.close();
    std::cout << "The currently reading file costs " << byteNum<< " bytes of memory.\n";

    std::ifstream bin;
    bin.open(filepath, std::ios::in | std::ios::binary);
    bin.read((char *)outside_array_pointer, byteNum);

}