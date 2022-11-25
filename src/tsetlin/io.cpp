#include "io.h"
using std::vector;
using std::string;

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

