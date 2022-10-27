#include "TsetlinMachine.h"

TsetlinMachine::TsetlinMachine( MachineArgs args):
_inputSize(args.inputSize),
_outputSize(args.outputSize),
_clausePerOutput(args.clausePerOutput),
_T(args.T),
_sLow(args.sLow), _sHigh(args.sHigh),
_dropoutRatio(args.dropoutRatio),
_myArgs(args)
{
    Automata::AutomataArgs aArgs;
    aArgs.clauseNum = _clausePerOutput;
    aArgs.dropoutRatio = _dropoutRatio;
    aArgs.inputSize = _inputSize;
    aArgs.sLow = _sLow;
    aArgs.sHigh = _sHigh;
    aArgs.T = _T;

    _response.resize(_outputSize, vector<int>(1,0));    // Set dummy zero response as placeholder.
    for (int i = 0; i < _outputSize; i++)
    {
        aArgs.no = i;
        Automata thisAutomata(aArgs,_sharedData,_response[i]);
        _automatas.push_back(thisAutomata);
    }
    
}

bool
TsetlinMachine::modelIntegrityCheck(model targetModel)
{
    bool isRightArgument =  (targetModel.modelArgs == _myArgs);
    return isRightArgument;
}

void
TsetlinMachine::importModel(model targetModel)
{
    if(!modelIntegrityCheck(targetModel))
    {
        std::cout<<"Your Tsetlin Machine model failed integrity check!"<<std::endl;
        throw; return;
    }
    for (int i = 0; i < _outputSize; i++)
    {
        _automatas[i].importModel(targetModel.automatas[i]);
    }
}

TsetlinMachine::model
TsetlinMachine::exportModel()
{
    TsetlinMachine::model result;
    result.modelArgs = _myArgs;
    result.automatas.resize(_outputSize,Automata::model());
    for (int i = 0; i < _outputSize; i++)
    {
        result.automatas[i] = _automatas[i].exportModel();
    }
    return result;
}

/// @brief Check the integrity of argument 'data'.
/// @param response Input unknown size 2D vector.
/// @return Result of integrity check procedure.
bool
TsetlinMachine::dataIntegrityCheck( const vector<vector<int>> data)
{
    bool isZeroSize = (data.size()==0);
    bool isCorrectLength = true;
    for (int i = 0; i < data.size(); i++)
    {
        isCorrectLength &= (data[i].size() == _inputSize);
        if(!isCorrectLength)break;
    }
    bool result = (!isZeroSize) && (isCorrectLength);
    if (!result)
    {
        std::cout<<"Data failed integrity check."<<std::endl;
    }
    
    return result;
}

/// @brief Check the integrity of argument 'response'.
/// @param response Input unknown size 2D vector.
/// @return Result of integrity check procedure.
bool
TsetlinMachine::responseIntegrityCheck(const vector<vector<int>> response)
{
    bool isZeroSize = (response.size()==0);
    bool isCorrectLength = (response.size()==_outputSize);

    bool result = (!isZeroSize) && (isCorrectLength); 
    if (!result)
    {
        std::cout<<"Result failed integrity check."<<std::endl;
    }
    return result;
}

/// @brief Load data and train this Tsetlin machine.
/// @param data 2D vector shaped in ( sampleNum * _inputSize )
/// @param response 2D vector shaped in ( sampleNum * _outputSize )
/// @param epoch Max count of repeat training time.
void
TsetlinMachine::loadAndTrain(   vector<vector<int>> data,
                                vector<vector<int>> response,
                                int epoch)
{
    if( !dataIntegrityCheck(data) || 
        !responseIntegrityCheck(response)) throw;
    
    _sharedData = data;
    _response = response;

    for (int i = 0; i < epoch; i++)
    {
        for (int j = 0; j < _outputSize; j++)   // Each output corresponds an automata.
        {
            _automatas[j].learn();
        }
    }
}

/// @brief Load data and predict response using trained tsetlin machine.
/// @param data 2D vector shaped in ( sampleNum * _inputSize )
/// @return 2D vector shaped in ( sampleNum * _outputSize )
vector<vector<int>>
TsetlinMachine::loadAndPredict(vector<vector<int>> data)
{
    if( !dataIntegrityCheck(data)) throw;
    vector<vector<Automata::Prediction>> prediction(_outputSize,
                                                    vector<Automata::Prediction>(data.size(),Automata::Prediction()));
    _sharedData = data;
    for (int i = 0; i < _outputSize; i++)
    {
        prediction[i] = _automatas[i].predict(data);
    }
    vector<vector<int>> result(data.size(), vector<int>(_outputSize,0));
    for (int sampleIdx = 0; sampleIdx < data.size(); sampleIdx++)
    {
        double  maxConfidence = 0;
        int     competitorIdx = 0;
        for (int featureIdx = 0; featureIdx < _outputSize; featureIdx++)
        {
            if(prediction[featureIdx][sampleIdx].confidence > maxConfidence)
            {
                competitorIdx = featureIdx;
                maxConfidence = prediction[featureIdx][sampleIdx].confidence;
            }
        }
        result[sampleIdx][competitorIdx] = 1;
    }
    return result;
}