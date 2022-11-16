#include "TsetlinMachine.h"
#include "io.h"
using std::vector;

int main(int argc, char const *argv[])
{
    std::mt19937                    rng(std::random_device{}());
    int                             train_data_size;
    int                             test_data_size;
    int                             output_size;
    int                             input_size = 84;
    int                             clausePerOutput = 500;      // Only represent the number of clauses that have same polarity.
    double                          dropoutRatio = 0.5;

    double trainRatio = 0.9;
    int classNum = 2;output_size = classNum;
    vector<string> seqs = readcsvline<string>("/home/data/siRNA/e2sall/e2sIncSeqs.csv");
    vector<double> res = readcsvline<double>("/home/data/siRNA/e2sall/e2sIncResponse.csv");
    dataset data = prepareData(seqs,res,trainRatio,classNum);
    vector<vector<int>> train_seqs = data.trainData; train_data_size = train_seqs.size();
    vector<vector<int>> train_scores= data.trainResponse;
    vector<vector<int>> test_seqs = data.testData;  test_data_size = test_seqs.size();
    vector<vector<int>> test_scores= data.testResponse;
    /*
    vector<vector<int>>   train_seqs(train_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   train_scores(train_data_size, vector(output_size, 0));
    vector<vector<int>>   test_seqs(test_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   test_scores(test_data_size, vector(output_size, 0));
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_training_seq.csv", train_seqs);
    encodeHueskenScores("../data/siRNA/e2s/e2s_training_efficiency.csv", train_scores);
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_test_seq.csv", test_seqs);
    encodeHueskenScores("../data/siRNA/e2s/e2s_test_efficiency.csv", test_scores);
*/
    TsetlinMachine::MachineArgs mArgs;
    mArgs.clausePerOutput = clausePerOutput;
    mArgs.dropoutRatio = dropoutRatio;
    mArgs.inputSize = input_size;
    mArgs.outputSize = output_size;
    mArgs.sLow = 2.0f;
    mArgs.sHigh = 100.0f;
    mArgs.T = 1000;
    TsetlinMachine tm(mArgs);
    
    tm.load(train_seqs,train_scores);
    for (int i = 0; i < 20; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        tm.train(1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout <<"### Epoch"<<i<< " training"<<" consumes : " << diff.count() << " s\n";
    }
    
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
    auto precision = totalCorrect/(double) test_scores.size();
    std::cout<< "Precision:"<< precision<<std::endl;
    auto model = tm.exportModel();
    vector<string> ttag(4);
    ttag[0] = "low";
    ttag[1] = "high";
    //outputModelStat(model,precision,ttag,"/home/output/");
    outputModelStat(model,precision,ttag,"./");
    outputModelPattern(model, precision,ttag,"./");
    
    return 0;
}
