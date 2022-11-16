#include "TsetlinMachine.h"
#include "io.h"
#include <climits>
#include "AOAoptimizer.h"
using std::vector;
using std::string;

struct modelAndArgs
{
    double                  value;  // Necessary in RSA interface protocol
    int                     clauseNum;
    int                     T;
    TsetlinMachine::model   model;
    modelAndArgs()
    {
        value = -(__DBL_MAX__);
    }
    modelAndArgs(   double valueIn,
                    int TIn, int clauseN,
                    TsetlinMachine::model modelIn)
    {
        value = valueIn;
        clauseNum = clauseN;
        T = TIn;
        model = modelIn;
    }
};

struct tsetlinArgs
{
    double dropoutRatio;
    int inputSize;
    int outputSize;
    int epochNum;
    double sLow;
    double sHigh;
    dataset data;
    vector<int> vars;   // clausePerOutput and T become the variable.
    tsetlinArgs(){}
    tsetlinArgs( double dor, int is, int os,int epo, double sl, double sh, dataset indata)
    {
        epochNum = epo;
        dropoutRatio = dor;
        inputSize = is;
        outputSize = os;
        sLow = sl;
        sHigh = sh;
        data = indata;
        vars.resize(2,0);
    }
};


modelAndArgs siRNAdemo(tsetlinArgs funcArgs)
{
    vector<vector<int>> train_seqs = funcArgs.data.trainData;
    vector<vector<int>> train_scores= funcArgs.data.trainResponse;
    vector<vector<int>> test_seqs = funcArgs.data.testData;
    vector<vector<int>> test_scores= funcArgs.data.testResponse;
    int test_data_size = test_seqs.size();


    TsetlinMachine::model       bestModel;
    double                      bestPrecision = 0;
    TsetlinMachine::MachineArgs mArgs;
    mArgs.clausePerOutput = funcArgs.vars[0];
    mArgs.T = funcArgs.vars[1];
    mArgs.dropoutRatio = funcArgs.dropoutRatio;
    mArgs.inputSize = funcArgs.inputSize;
    mArgs.outputSize = funcArgs.outputSize;
    mArgs.sLow = funcArgs.sLow;
    mArgs.sHigh = funcArgs.sHigh;
    TsetlinMachine tm(mArgs);
    tm.load(train_seqs,train_scores);
    for (int i = 0; i < funcArgs.epochNum; i++)
    {
        tm.train(1);
        vector<vector<int>> predict = tm.loadAndPredict(test_seqs);
        int totalCorrect = 0;
        for (int sample = 0; sample < predict.size(); sample++)
        {
            bool correct = true;
            for (int digit = 0; digit < predict[0].size(); digit++)
            {
                if(predict[sample][digit] != test_scores[sample][digit])
                {
                    correct = false;
                    break;
                }
            }
            totalCorrect += correct;
        }
        double thisPrecision = totalCorrect / (double)test_data_size;
        if(thisPrecision>bestPrecision)
        {
            bestPrecision = thisPrecision;
            bestModel = tm.exportModel();
        }
    }
    modelAndArgs result(bestPrecision,mArgs.T,mArgs.clausePerOutput,bestModel);
    return result;
}

int main(int argc, char const *argv[])
{
    double trainRatio = 0.9;
    int classNum = 2;
    vector<string> seqs = readcsvline<string>("/home/data/siRNA/e2sall/e2sIncSeqs.csv");
    vector<double> res = readcsvline<double>("/home/data/siRNA/e2sall/e2sIncResponse.csv");
    dataset data = prepareData(seqs,res,trainRatio,classNum);
    // Tsetlin Machine common arguments.
    int             inputSize= 84;
    int             outputSize= 2;
    int             epochNum = 60;
    double          dropoutRatio = 0.3;
    tsetlinArgs     funcArgs(dropoutRatio,inputSize,outputSize,epochNum,4.0f,200.0f,data);

    AOAoptimizer<modelAndArgs, tsetlinArgs, int>::args arg;
    arg.dimensionNum = 2;
    arg.optimizerNum= 94;
    arg.evaluateFunc = siRNAdemo;
    arg.gFuncArgs = funcArgs;
    arg.iterNum= 100;
    arg.lowerBounds= vector<int>{100,50};
    arg.upperBounds= vector<int>{500, 5000};
    AOAoptimizer<modelAndArgs, tsetlinArgs, int> env(arg);
    modelAndArgs result = env.optimize();
    std::cout<<result.value<<std::endl;
    //modelOutput(result.model,result.value,"/home/output/");    // Last argument is up to you.
    vector<string> ttag(2);
    ttag[0] = "low";
    ttag[1] = "high";
    outputModelStat(result.model,result.value,ttag,"/home/output/");
    return 0;
}