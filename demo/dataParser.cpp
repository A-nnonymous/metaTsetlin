#include "io.h"
#include "nucleotides.h"
using std::vector;
using std::string;
int main()
{
    int                             train_data_size = 1229;
    int                             test_data_size = 139;
    int                             input_size = 84;

    vector<vector<int>>   train_seqs(train_data_size, vector<int>(input_size, 0));
    vector<vector<int>>   test_seqs(test_data_size, vector<int>(input_size, 0));
    vector<vector<double>>   train(train_data_size,vector<double>(input_size + 1, 0));
    vector<vector<double>>   test(test_data_size,vector<double>(input_size + 1, 0));
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_training_seq.csv", train_seqs);
    encodeHueskenSeqs("../data/siRNA/e2s/e2s_test_seq.csv", test_seqs);

    // Concatenate scores
    std::ifstream train_score ("/home/metaTsetlin/data/siRNA/e2s/e2s_training_efficiency.csv");
    std::ifstream test_score("/home/metaTsetlin/data/siRNA/e2s/e2s_test_efficiency.csv");
    std::string score_string;
    float this_score;
    int idx = 0;
    while (std::getline (train_score, score_string))
    {
        this_score = ::atof (score_string.c_str ());
        train[idx++][input_size] = this_score;
    }
    idx = 0;
    while (std::getline (test_score, score_string))
    {
        this_score = ::atof (score_string.c_str ());
        test[idx++][input_size] = this_score;
    }
    for(int i = 0; i < train_data_size ; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            train[i][j] = train_seqs[i][j]==0 ? 0.0f : 1.0f;
        }
    }
    for(int i = 0; i < test_data_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            test[i][j] = test_seqs[i][j]==0 ? 0.0f : 1.0f;
        }
    }
    vector<string> header(input_size + 1);
    for (int i = 0; i < input_size; i++)
    {
        string tag;
        int position = i/4;
        switch (i%4)
        {
        case 0:
                tag = "G" + std::to_string(position);
            break;
        case 1:
                tag = "C" + std::to_string(position);
            break;
        case 2:
                if((input_size - i)/8)[[likely]] // not the last two nucleotides
                {
                    tag = "U"+ std::to_string(position);
                }
                else
                {
                    tag = "T"+ std::to_string(position);
                }
            break;
        case 3:
                tag = "A"+ std::to_string(position);
            break;
        default:
            break;
        }
        header[i] = tag;
    }
    header[input_size] = "score";
    
    //write_csv<int>(train, train_data_size,input_size+1,true, header, "./train");
    //write_csv<int>(test, test_data_size,input_size+1,true, header, "./test");

}