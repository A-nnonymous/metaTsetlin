#include "Clause.h"
using std::vector;

/// @brief A tsetlin automata is fundamental object to learn a digit of output from input.
class Automata{
public:
    struct AutomataArgs
    {
        int     no;
        int     inputSize;
        int     clauseNum;
        int     T;
        double  sLow, sHigh;
        double  dropoutRatio;
    };
    struct Prediction
    {
        int     result;
        double  confidence;
        Prediction()
        {
            result = 0;
            confidence = 0;
        }
    };
    struct model
    {
        int                     no;
        vector<Clause::model>   positiveClauses;
        vector<Clause::model>   negativeClauses;
        model(){}
    };
    

private:
    const int                   _no;
    const int                   _clauseNum;         // Attention, this is only represent one single polarity.
    const int                   _T;
    const double                _sLow;
    const double                _sHigh;             // This is for multigranular clauses.
    const double                _dropoutRatio;      // Random dropout some clauses.
    vector<vector<int>>         &_sharedInputData;  // When start traning, reference dataset from TM.
    vector<int>                 &_targets;

    std::mt19937                _rng;               // Shared random number generator using Mersenne twister.
    int                         _voteSum;           // Sum of all clauses' vote.
    vector<Clause>              _positiveClauses;
    vector<Clause>              _negativeClauses;

    int     forward(vector<int> &datavec);
    void    backward(int response);
    bool    modelIntegrityCheck(model targetModel);
public:
    Automata(AutomataArgs args, vector<vector<int>> &input, vector<int> &target);

    void                importModel(model targetModel);
    model               exportModel();
    void                learn();
    vector<Prediction>  predict(vector<vector<int>> &input);
};