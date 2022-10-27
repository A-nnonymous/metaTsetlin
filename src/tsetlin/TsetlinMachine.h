#include "Automata.h"
using std::vector;

class TsetlinMachine{
public:
    struct MachineArgs
    {
        int     inputSize;
        int     outputSize;
        int     clausePerOutput;
        int     T;
        double  sLow, sHigh;
        double  dropoutRatio;

        bool operator==(MachineArgs a)const
        {
            return  (a.clausePerOutput = this->clausePerOutput) &&
                    (a.dropoutRatio = this->dropoutRatio) &&
                    (a.inputSize = this->inputSize) &&
                    (a.outputSize = this->outputSize)&&
                    (a.sHigh = this->sHigh)&&
                    (a.sLow = this->sLow) &&
                    (a.T = this->T);
        }
    };
    struct model
    {
        MachineArgs             modelArgs;
        vector<Automata::model> automatas;
    };
    

private:
    const int               _inputSize;
    const int               _outputSize;
    const int               _clausePerOutput;
    const int               _T;
    const double            _sLow, _sHigh;
    const double            _dropoutRatio;
    const MachineArgs       _myArgs;

    vector<Automata>        _automatas;
    vector<vector<int>>     _sharedData;
    vector<vector<int>>     _response;      // Each row is a reflection of multi-dimensional dataset.

    bool    dataIntegrityCheck( const vector<vector<int>> data);
    bool    responseIntegrityCheck(const vector<vector<int>> response);
    bool    modelIntegrityCheck(model targetModel);

public:
    TsetlinMachine( MachineArgs args);

    void                importModel(model targetModel);
    model               exportModel();
    void                loadAndTrain(   vector<vector<int>> data,
                                        vector<vector<int>> response,
                                        int epoch);
    vector<vector<int>> loadAndPredict( vector<vector<int>> data);
};