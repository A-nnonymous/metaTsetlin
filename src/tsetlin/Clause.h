#include <vector>
#include <iostream>
#include <chrono>
#include <random>

using std::vector;

/// @brief This clause use integer as literal as default.
class Clause{
public:
    struct ClauseArgs
    {
        int     no;         // Unique tag of all clauses.
        bool    isPositiveClause;
        int     inputSize;
        double  specificity;
    };
    struct model
    {
        int         no;
        vector<int> positiveLiteral;
        vector<int> negativeLiteral;
        model(){}
    };

private:
    const int           _no;
    const bool          _isPositiveClause;
    const int           _literalNum;
    const double        _s,_sInv,_sInvConj;     // Granular parameter passed from upper class, can be multigranular.

    std::mt19937        _rng;
    vector<int>         _positiveLiterals;
    vector<int>         _negativeLiterals;
    vector<bool>        _posInclusionMask;
    vector<bool>        _negInclusionMask;
    vector<bool>        _inputMask;

    int                 _vote;
    bool                _isVoteDirty;

    inline bool modelIntegrityCheck(model targetModel);

public:
    Clause(ClauseArgs args);

    void    importModel(model targetModel);
    model   exportModel();
    int     vote(vector<int> in);
    void    feedbackTypeI();
    void    feedbackTypeII();

};