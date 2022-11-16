#include "TsetlinMachine.h"
#include "io.h"
#include <iostream>
#include <time.h>
using std::vector;
vector<vector<int>> transpose(vector<vector<int>> original)
{
    int rowNum = original.size();
    int colNum = original[0].size();
    vector<vector<int>> result(colNum, vector<int>(rowNum,0));
    for (int row = 0; row < rowNum; row++)
    {
        for (int col = 0; col < colNum; col++)
        {
            result[col][row] = original[row][col];
        }
    }
    return result;
}
int main() {
    std::mt19937                    rng(time(nullptr));
    vector<vector<int>>   inputs = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    vector<vector<int>>   outputs = {
        { 0, 1 },
        { 1, 0 },
        { 1, 0 },
        { 0, 1 },
    };
    TsetlinMachine::MachineArgs mArgs;
    mArgs.clausePerOutput = 10;
    mArgs.dropoutRatio = 0.5;
    mArgs.inputSize = 2;
    mArgs.outputSize = 2;
    mArgs.sLow = 4.0f;
    mArgs.sHigh = 4.0f;
    mArgs.T = 4;
    TsetlinMachine tm(mArgs);
    
    tm.load(inputs, outputs);
    
    vector<vector<int>> prediction;
    clock_t start = clock();
    for (int i = 0; i < 10000; i++)
    {
        tm.train(1);
    }
    clock_t duration = clock() - start;
    
    std::cout<< duration/(double)CLOCKS_PER_SEC <<std::endl;
    prediction = tm.loadAndPredict(inputs);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            std::cout<<prediction[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }
    /*
    std::cout<<"This tm's job is done, saving model"<<std::endl;
    TsetlinMachine::model temp = tm.exportModel();
    std::cout<<"model size is :" << sizeof(temp)<<std::endl;
    saveModel(temp,"./xor.tm");
    */
    /*
    TsetlinMachine::model receive = loadModel("./xor.tm");
    TsetlinMachine tm2(receive);

    std::cout<<"Loaded model's prediction is:"<<std::endl;
    vector<vector<int>> prediction = tm2.loadAndPredict(inputs);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            std::cout<<prediction[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }

*/
    return 0;
}
