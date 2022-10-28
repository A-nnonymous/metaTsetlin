#include "TsetlinMachine.h"
#include "io.h"
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

int main(int argc, char const *argv[])
{
    std::mt19937                    rng(std::random_device{}());
    int                             train_data_size = 1229;
    int                             test_data_size = 139;
    int                             output_size = 4;
    int                             input_size = 84;
    int                             clausePerOutput = 1000;      // Only represent the number of clauses that have same polarity.
    double                          dropoutRatio = 0.0;

    vector<vector<int>>   train_seqs(train_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   train_scores(train_data_size, vector(output_size, 0));
    vector<vector<int>>   test_seqs(test_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   test_scores(test_data_size, vector(output_size, 0));
    parse_huesken_seqs("/home/data/siRNA/e2s/e2s_training_seq.csv", train_seqs);
    parse_huesken_scores("/home/data/siRNA/e2s/e2s_training_efficiency.csv", train_scores);
    parse_huesken_seqs("/home/data/siRNA/e2s/e2s_test_seq.csv", test_seqs);
    parse_huesken_scores("/home/data/siRNA/e2s/e2s_test_efficiency.csv", test_scores);
    TsetlinMachine::MachineArgs mArgs;
    mArgs.clausePerOutput = clausePerOutput;
    mArgs.dropoutRatio = dropoutRatio;
    mArgs.inputSize = input_size;
    mArgs.outputSize = output_size;
    mArgs.sLow = 10.0f;
    mArgs.sHigh = 20.0f;
    mArgs.T = 400;
    TsetlinMachine tm(mArgs);
    
    tm.load(train_seqs,train_scores);
    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        tm.train(1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Training"<<" consumes : " << diff.count() << " s\n";
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
    std::cout<< "Precision:"<< totalCorrect/(double) test_scores.size()<<std::endl;
    
    
    return 0;
}
