#include "TsetlinMachine.h"
#include "io.h"
#include "nucleotides.h"
using std::vector;

int main(int argc, char const *argv[])
{
    int                             train_data_size;
    int                             test_data_size;
    int                             output_size;
    int                             clausePerOutput = 500;      // Only represent the number of clauses that have same polarity.
    double                          dropoutRatio = 0.5;

    int classNum = 2;
    int epochNum = 10;
    double trainRatio = 0.9;
    output_size = classNum;
    nucTransformer transformer;
    vector<string> seqs = readcsvline<string>("../data/siRNA/e2sall/e2sIncSeqs.csv");
    vector<double> res = readcsvline<double>("../data/siRNA/e2sall/e2sIncResponse.csv");
    dataset data = transformer.parseAndDivide(seqs,res,trainRatio,classNum);
    vector<vector<int>> train_seqs = data.trainData; train_data_size = train_seqs.size();
    vector<vector<int>> train_scores= data.trainResponse;
    vector<vector<int>> test_seqs = data.testData;  test_data_size = test_seqs.size();
    vector<vector<int>> test_scores= data.testResponse;
    vector<double> responseThreshold = data.responseThreshold;
    vector<string> tierTags = data.tierTags;
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
    mArgs.inputSize = train_seqs[0].size();
    std::cout<< "training data consume "<<mArgs.inputSize << " int32 each"<<std::endl;
    mArgs.outputSize = output_size;
    mArgs.sLow = 2.0f;
    mArgs.sHigh = 100.0f;
    mArgs.T = 500;
    TsetlinMachine tm(mArgs,tierTags);
    
    tm.load(train_seqs,train_scores);
    for (int i = 0; i < epochNum; i++)
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
    vector<string> headers{"NUC","NUCvoice","GC","GCvoice"};
    transformer.deparseAndOutput(model,precision,headers,"./");
    return 0;
}
