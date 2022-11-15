#include "TsetlinMachine.h"
#include "io.h"
#include <climits>
#include "RSAoptimizer.h"
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
    dataset data;
    vector<int> vars;   // clausePerOutput and T become the variable.
    tsetlinArgs(){}
    tsetlinArgs( double dor, int is, int os,int epo, double sl, double sh ,dataset indata)
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
    /*
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
    */
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
    double          trainRatio = 0.9;
    int             classNum = 2;
    vector<string>  seqs = readcsvline<string>("/home/data/siRNA/e2sall/e2sIncSeqs.csv");
    vector<double>  res = readcsvline<double>("/home/data/siRNA/e2sall/e2sIncResponse.csv");
    dataset data = prepareData(seqs,res,trainRatio,classNum);
    // Tsetlin Machine common arguments.
    int             inputSize= data.testData[0].size();
    int             outputSize= classNum;
    int             epochNum = 60;
    double          dropoutRatio = 0.5;
    tsetlinArgs     funcArgs(dropoutRatio,inputSize,outputSize,epochNum,2.0f,50.0f, data);

    // RSA algorithm arguments;
    int             N = 94;     // Number of individual optimizer.
    int             dimNum = 2;
    int             maxIter = 100;
    double          alpha = 0.1;
    double          beta = 0.005;

    vector<int> mins{100, 50};
    vector<int> maxes{500, 5000};
    auto limits = Predator<modelAndArgs,tsetlinArgs,int>::rangeLimits(mins,maxes);
    auto searchArgs = Predator<modelAndArgs,tsetlinArgs,int>::searchArgs(dimNum,maxIter,alpha,beta,limits);
    
    auto RSA = RSAoptimizer<modelAndArgs,tsetlinArgs,int>(N, siRNAdemo,funcArgs,searchArgs);
    modelAndArgs result;
    result = RSA.optimize();
    //modelOutput(result.model,result.value,"/home/output/");    // Last argument is up to you.
    vector<string> ttag(2);
    ttag[0] = "low";
    ttag[1] = "high";
    modelOutputStat(result.model,result.value,ttag,"/home/output/");
    return 0;
}