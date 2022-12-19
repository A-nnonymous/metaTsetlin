#include "TsetlinMachine.h"
#include "io.h"
#include "nucleotides.h"
#include <climits>
#include "AOAoptimizer.h"
#include "RSAoptimizer.h"
#include "PSOoptimizer.h"

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


modelAndArgs siRNAdemo(tsetlinArgs &funcArgs)
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
    
    TsetlinMachine tm(mArgs,funcArgs.data.tierTags);
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
    nucTransformer  transformer;
    ////////// Define range of mutable parameters that to be optimized ///////////
    int         optimizeEpoch = 20;
    int         threadNum = 94;
    vector<int> mins{100, 50};
    vector<int> maxes{500, 5000};
    ////////// Define range of mutable parameters that to be optimized ///////////

    ////////////// Tsetlin Machine parameters initialization///////////////
    int             responseClassNum = 2;
    int             outputSize= responseClassNum;
    int             epochNum = 50;
    double          dropoutRatio = 0.5;
    double          trainRatio = 0.9;
    vector<string>  seqs = readcsvline<string>("../data/siRNA/e2sall/e2sIncSeqs.csv");
    vector<double>  res = readcsvline<double>("../data/siRNA/e2sall/e2sIncResponse.csv");

    dataset         data = transformer.parseAndDivide(seqs,res,trainRatio,responseClassNum);
    int             inputSize= data.trainData[0].size();
    tsetlinArgs     funcArgs(dropoutRatio,inputSize,outputSize,epochNum,2.0f,200.0f,data);
    ////////////// Tsetlin Machine parameters initialization ///////////////


    ////////////// AOA algorithm parameters initialization ///////////
    AOAoptimizer<modelAndArgs, tsetlinArgs, int>::args AOAarg;
    AOAarg.dimensionNum = 2;
    AOAarg.optimizerNum= threadNum;
    AOAarg.evaluateFunc = siRNAdemo;
    AOAarg.gFuncArgs = funcArgs;
    AOAarg.iterNum= optimizeEpoch;
    AOAarg.lowerBounds= mins;
    AOAarg.upperBounds= maxes;
    ////////////// AOA algorithm parameters initialization ///////////

    ////////////// RSA algorithm parameters initialization ///////////
    int             N = threadNum;     // Number of individual optimizer.
    int             dimNum = 2;
    int             maxIter = optimizeEpoch;
    double          alpha = 0.1;
    double          beta = 0.005;
    auto limits = Predator<modelAndArgs,tsetlinArgs,int>::rangeLimits(mins,maxes);
    auto searchArgs = Predator<modelAndArgs,tsetlinArgs,int>::searchArgs(dimNum,maxIter,alpha,beta,limits);
    ////////////// RSA algorithm parameters initialization ///////////

    ////////////// PSO algorithm parameters initialization ///////////
    PSOoptimizer<modelAndArgs, tsetlinArgs, int>::args PSOargs;
    PSOargs.dimension = 2;
    PSOargs.dt = 0.001;
    PSOargs.ego = 2;
    PSOargs.evaluateFunc = siRNAdemo;
    PSOargs.gFuncArgs = funcArgs;
    PSOargs.maxIter = optimizeEpoch;
    PSOargs.omega = 0.9;
    PSOargs.particleNum = threadNum;
    PSOargs.rangeMax = mins;
    PSOargs.rangeMin = maxes;
    PSOargs.vMax = vector<int>{100,100};
    PSOargs.vMin = vector<int>{0,0};
    ////////////// PSO algorithm parameters initialization ///////////

    /////////////////// Main computing process /////////////////
    PSOoptimizer<modelAndArgs, tsetlinArgs, int> PSO(PSOargs);
    AOAoptimizer<modelAndArgs, tsetlinArgs, int> AOA(AOAarg);
    RSAoptimizer<modelAndArgs, tsetlinArgs, int> RSA(N, siRNAdemo,funcArgs,searchArgs);
    vector<modelAndArgs> result{AOA.optimize(), RSA.optimize(), PSO.optimize()};
    vector<string> bestOptimizerName{"AOA","RSA","PSO"};
    /////////////////// Main computing process /////////////////

    /////////////////// Result output //////////////////////////
    double bestValue = -1;
    auto bestIdx = -1;
    modelAndArgs bestModel;
    for (int i = 0; i < 3; i++)
    {
        if(result[i].value > bestValue)
        {
            bestValue = result[i].value;
            bestIdx = i;
        }
    }
    std::cout<<"Prediction precision is optimized to: "<<result[bestIdx].value
                <<" by "+bestOptimizerName[bestIdx] + "optimizer."<<std::endl;
    vector<string> deparseHeaders{  "Nucleotides pattern",
                                    "NucPattern strength",
                                    "GC Content Pattern",
                                    "GCpattern strength"};
    transformer.deparseAndOutput(result[bestIdx].model,result[bestIdx].value,deparseHeaders,"./");
    /////////////////// Result output //////////////////////////
    return 0;
}