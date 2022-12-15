#include "Optimizer.h"
#include <chrono>

using std::vector;
using std::thread;

template<typename output, typename funcArgs, typename rangeDtype>
class AOAoptimizer{
public:
    struct args
    {
        int                     optimizerNum;
        int                     dimensionNum;
        int                     iterNum;
        output                  (*evaluateFunc)(funcArgs&);
        funcArgs                gFuncArgs;
        vector<rangeDtype>      lowerBounds;
        vector<rangeDtype>      upperBounds;
    };
    

private:
    static inline const double      Alpha = 0.5;
    const int                       _optimizerNum;
    const int                       _iterNum;

    double                          _w;
    double                          _MOP;

    vector<
        Optimizer
            < 
            output, 
            funcArgs, 
            rangeDtype 
            > 
        >                           _optimizers;
        
    double                          _gbestValue;
    vector<rangeDtype>              _gbestPosition;
    double                          _gworstValue;
    vector<rangeDtype>              _gworstPosition;
    output                          _gbestProperty;

    void exploitation()
    {
        vector<thread> threadPool;
        ////////////////////// Multithread evaluating//////////////////////
        for (int i = 0; i < _optimizerNum; i++)
        {
            thread th(&Optimizer<output,funcArgs,rangeDtype>::exploit, &_optimizers[i]);
            threadPool.push_back(std::move(th));
        }
        for (int i = 0; i < _optimizerNum; i++)
        {
            threadPool[i].join();
        }
        ////////////////////// Multithread evaluating//////////////////////
        int bestCandidateIdx = 0;
        int worstCandidateIdx = 0;
        bool touchedBest = false;
        bool touchedWorst = false;
        for (int i = 0; i < _optimizerNum; i++)
        {
            double thisValue = _optimizers[i].getValue();
            if(thisValue > _gbestValue)
            {
                _gbestValue = thisValue;
                bestCandidateIdx = i;
                touchedBest = true;
            }
            if(thisValue < _gworstValue)
            {
                _gworstValue = thisValue;
                worstCandidateIdx = i;
                touchedWorst = true;
            }
        }
        if(touchedBest)
        {
            _gbestProperty = _optimizers[bestCandidateIdx].getProperty();
            _gbestPosition = _optimizers[bestCandidateIdx].getPosition();
        }
        if(touchedWorst)
        {
            _gworstPosition = _optimizers[worstCandidateIdx].getPosition();
        }
    }

    void exploration()
    {
        for (int i = 0; i < _optimizerNum; i++)
        {
            _optimizers[i].explore();
        }
        
    }

public:
    AOAoptimizer(args thisArg):
    _optimizerNum(thisArg.optimizerNum),
    _iterNum(thisArg.iterNum)
    {
        _gbestProperty = output(); 
        _gbestValue = -(__DBL_MAX__); 
        _gworstValue = (__DBL_MAX__); 
        _gbestPosition.resize(thisArg.dimensionNum);
        _gworstPosition.resize(thisArg.dimensionNum);
        typename Optimizer<output,funcArgs,rangeDtype>::shared 
        shr(_MOP,_w,_gbestValue,_gworstValue,_gbestPosition,_gworstPosition);

        typename Optimizer<output,funcArgs,rangeDtype>::limit lim;
        lim.min = thisArg.lowerBounds;
        lim.max = thisArg.upperBounds;

        typename Optimizer<output,funcArgs,rangeDtype>::args oArg;
        oArg.dimensionNum = thisArg.dimensionNum;
        oArg.lim = lim;
        oArg.evaluateFunc = thisArg.evaluateFunc;
        oArg.gFuncArgs = thisArg.gFuncArgs;

        for (int i = 0; i < _optimizerNum; i++)
        {
            Optimizer<output,funcArgs,rangeDtype> o(oArg, shr);
            _optimizers.push_back(o);
        }
    }

    output optimize()
    {
        std::cout<< "\n############## AOA optimizer started #################"<<std::endl;
        output result = output();
        for (int i = 1; i <= _iterNum; i++)
        {
            _w = (-2/M_PI) * (atan(i)) + 1 + 0.5 * exp(-i/5);
            _MOP = 1 - (pow(i,(1 / Alpha)) / pow(_iterNum, (1 / Alpha)));
            auto start = std::chrono::high_resolution_clock::now();
            exploitation();
            exploration();
            std::cout<<"###Iter "<<i<< ", best value is optimized to:" <<_gbestProperty.value;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            int minuteETA = int((_iterNum-i)*diff.count()/60);
            int secETA = (diff.count()*(_iterNum-i) - minuteETA*60);
            std::cout <<", consumes : " << diff.count() << " s\t"
                        << "ETA: "<<minuteETA<<"min "
                                    <<secETA<<"s"<<std::endl;

        }
        result = _gbestProperty;

        std::cout<<"\nOptimization completed "<< ", best value is optimized to:" <<result.value<<std::endl;
        std::cout<<"Best args as below:" << std::endl;
        for (int i = 0; i < _gbestPosition.size(); i++)
        {
            std::cout<<"\tArgs["<< i<<"] = "<< _gbestPosition[i] <<std::endl;
        }
        std::cout<<std::endl;
        return result;
    }
};