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
    const int                   _inputSize;
    const int                   _clauseNum;         // Attention, this is only represent one single polarity.
    const int                   _T;
    const double                _sLow;
    const double                _sHigh;             // This is for multigranular clauses.
    const double                _dropoutRatio;      // Random dropout some clauses.
    vector<vector<__m512i>>     &_sharedInputData;  // When start traning, reference dataset from TM.
    vector<int>                 &_targets;

    pcg64_fast                  _rng;
    int                         _voteSum;           // Sum of all clauses' vote.
    vector<Clause>              _positiveClauses;
    vector<Clause>              _negativeClauses;

    int     forward(vector<__m512i> &datavec);
    void    backward(int &response);
    bool    modelIntegrityCheck(model targetModel);
public:
    Automata(AutomataArgs args, vector<vector<__m512i>> &input, vector<int> &target);

    void                learn();
    vector<Prediction>  predict(vector<vector<__m512i>> &input);

    model               exportModel();
    void                importModel(model &targetModel);
};