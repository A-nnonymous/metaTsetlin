#include "TsetlinMachine.h"
#include "io.h"
#include <climits>
#include "AOAoptimizer.h"
using std::vector;

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
    vector<int> vars;   // clausePerOutput and T become the variable.
    tsetlinArgs(){}
    tsetlinArgs( double dor, int is, int os,int epo, double sl, double sh )
    {
        epochNum = epo;
        dropoutRatio = dor;
        inputSize = is;
        outputSize = os;
        sLow = sl;
        sHigh = sh;
        vars.resize(2,0);
    }
};


modelAndArgs siRNAdemo(tsetlinArgs funcArgs)
{
    int                             train_data_size = 1229;
    int                             test_data_size = 139;
    double                          dropoutRatio = 0.5;
    int                             output_size = funcArgs.outputSize;
    int                             input_size = funcArgs.inputSize;

    vector<vector<int>>   train_seqs(train_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   train_scores(train_data_size, vector(output_size, 0));
    vector<vector<int>>   test_seqs(test_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   test_scores(test_data_size, vector(output_size, 0));
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_training_seq.csv", train_seqs);
    encodeHueskenScores("../data/siRNA/e2s/e2s_training_efficiency.csv", train_scores);
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_test_seq.csv", test_seqs);
    encodeHueskenScores("../data/siRNA/e2s/e2s_test_efficiency.csv", test_scores);

    TsetlinMachine::model       bestModel;
    double                      bestPrecision = 0;
    TsetlinMachine::MachineArgs mArgs;
    mArgs.clausePerOutput = funcArgs.vars[0];
    mArgs.T = funcArgs.vars[1];
    mArgs.dropoutRatio = dropoutRatio;
    mArgs.inputSize = input_size;
    mArgs.outputSize = output_size;
    mArgs.sLow = 2.0f;
    mArgs.sHigh = 100.0f;
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
    // Tsetlin Machine common arguments.
    int             inputSize= 84;
    int             outputSize= 4;
    int             epochNum = 60;
    double          dropoutRatio = 0.3;
    tsetlinArgs     funcArgs(dropoutRatio,inputSize,outputSize,epochNum,2.0f,100.0f);

    AOAoptimizer<modelAndArgs, tsetlinArgs, int>::args arg;
    arg.dimensionNum = 2;
    arg.optimizerNum= 94;
    arg.evaluateFunc = siRNAdemo;
    arg.gFuncArgs = funcArgs;
    arg.iterNum= 50;
    arg.lowerBounds= vector<int>{100,20};
    arg.upperBounds= vector<int>{500, 1500};
    AOAoptimizer<modelAndArgs, tsetlinArgs, int> env(arg);
    modelAndArgs result = env.optimize();
    std::cout<<result.value<<std::endl;
    modelOutput(result.model,result.value,"/home/output/");    // Last argument is up to you.
    return 0;
}