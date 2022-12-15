// An implementation of Tsetlin Machine in C++ using SIMD instructions and meta-heuristic optimizers.

// The MIT License (MIT)
// Copyright (c) 2022 Pan Zhaowu <panzhaowu21s@ict.ac.cn>

//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#pragma once
#include "Predator.h"
#include <thread>
#include <chrono>
using std::vector;
using std::thread;


/// @brief Orchestrator of this algorithm.
/// @tparam output Output struct of fitness function, which must contain fitness value named "value" initialized to (double)-Inf.
/// @tparam funcArgs Fitness function arguments(trivial or struct) passed from user.
/// @tparam rangeDtype Data type of search range.
template<typename output, typename funcArgs, typename rangeDtype>
class RSAoptimizer{
private:
    //std::mt19937                        _rng;
    pcg64_fast                          _rng;
    const int                           _predatorNum;
    const funcArgs                      _gFuncArgs;
    const typename Predator
                    <
                    output,
                    funcArgs,
                    rangeDtype
                    >
                    ::searchArgs        _gSearchArgs;

    vector<Predator
                <
                output,
                funcArgs,
                rangeDtype
                >
            >                           _predators;
    vector<vector<rangeDtype>>          _allPosition;
    vector<rangeDtype>                  _gbPosition;
    output                              _gbProperty;                // output all data returned from best predator.
    output                              (*_funcPtr)(funcArgs&);      // Fitness function pointer
    int                                 _iterCounter;


    /// @brief Evalution of all predators, have side effect on global optima and its property.
    void exploitation()
    {
        /////////////////////////// Multithread evaluation////////////////////////////
        static vector<thread> threadPool;
        for(auto    thisPredator = _predators.begin();
                    thisPredator != _predators.end();
                    thisPredator ++ )
        {
            thread th(&Predator<output,funcArgs,rangeDtype>::exploit,thisPredator);
            threadPool.push_back(move(th));
        }
        for(auto &thread: threadPool)   thread.join();
        threadPool.clear();
        /////////////////////////// Multithread evaluation////////////////////////////
        int candidateIdx = 0, predatorIdx = 0;
        bool touched = false;
        for(auto    thisPredator = _predators.begin();
                    thisPredator != _predators.end();
                    thisPredator ++ )
        {
            if(thisPredator->getValue() > _gbProperty.value)
            {
                candidateIdx = predatorIdx;
                touched = true;
            }
            predatorIdx++;
        }
        if(touched)
        {
            _gbProperty = _predators[candidateIdx].getProperty();
            _gbPosition = _allPosition[candidateIdx];
        }

        
    }

    void exploration()
    {
        static std::uniform_real_distribution<double>       evoScaleDist(-2,2);
        double evoSense = evoScaleDist(_rng)*_iterCounter/(double)_gSearchArgs.maxIter;
        /////////////////////////// Multithread evaluation////////////////////////////
        static vector<thread> threadPool;
        for(auto    thisPredator = _predators.begin();
                    thisPredator != _predators.end();
                    thisPredator ++ )
        {
            thread th(&Predator<output,funcArgs,rangeDtype>::explore,
                            thisPredator,
                            _gbPosition,
                            evoSense,
                            _iterCounter);
            threadPool.push_back(move(th));
        }
        for(auto &thread: threadPool)   thread.join();
        threadPool.clear();
        /////////////////////////// Multithread evaluation////////////////////////////
    }

public:
    /// @brief Constructor of hibitat.
    /// @param predatorNum Number of individual optimizer.
    /// @param fitnessFunc Fitness function used to evaluate arguments.
    /// @param gFuncArgs Fitness function related arguments passed from user, must include vector named 'vars' stand for changable parameters.
    /// @param gSearchArgs Optimizer related arguments passed from user.
    RSAoptimizer(int                                                             predatorNum,
            output                                                          (*fitnessFunc)(funcArgs&),
            funcArgs                                                        gFuncArgs,
            typename Predator<output,funcArgs,rangeDtype>::searchArgs       gSearchArgs)
    :
    _predatorNum(predatorNum),
    _gSearchArgs(gSearchArgs),
    _gFuncArgs(gFuncArgs)
    {
        //_rng = std::mt19937(std::random_device{}());
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg64_fast _rng(seed_source);
        _funcPtr = fitnessFunc;
        _gbProperty = output();       // Initialize output property to default.
        _allPosition.resize(_predatorNum,
                            vector<rangeDtype>(_gSearchArgs.dimension, 0));

        for (int dim = 0; dim < _gSearchArgs.dimension; dim++)
        {
            rangeDtype thisDimMin = _gSearchArgs.searchRange.minLimits[dim];
            rangeDtype thisDimMax = _gSearchArgs.searchRange.maxLimits[dim];
            std::uniform_real_distribution<double>  disti(thisDimMin, thisDimMax);
            // If rangeDtype is integer, then the limit range is semi-closed like in python.
            for(int pred = 0; pred < predatorNum; pred++)
            {
                _allPosition[pred][dim] = (rangeDtype) disti(_rng);
            }
        }
        
        // Instantiate all predators, passing i as ID.
        for (int i = 0; i < _predatorNum; i++)
        {
            _predators.push_back(Predator<output,funcArgs,rangeDtype>(  _funcPtr,
                                                                        _gSearchArgs,
                                                                        _gFuncArgs,
                                                                        _allPosition,
                                                                        i));
        }
    }

    /// @brief  Main optimize function of RSA algorithm
    output optimize()
    {
        std::cout<< "############## RSA optimizer started #################"<<std::endl;
        for(_iterCounter=1; _iterCounter<=_gSearchArgs.maxIter; _iterCounter++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            exploitation();
            exploration();
            std::cout<<"###Iter "<<_iterCounter<< ", best value is optimized to:" <<_gbProperty.value;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            int minuteETA = int((_gSearchArgs.maxIter-_iterCounter)*diff.count()/60);
            int secETA = (diff.count()*(_gSearchArgs.maxIter-_iterCounter) - minuteETA*60);
            std::cout <<", consumes : " << diff.count() << " s\tETA: "<<minuteETA<<"min "
                                    <<secETA<<"s"<<std::endl;

        }
        auto result = _gbProperty;
        std::cout<<"\nOptimization completed "<< ", best value is optimized to:" <<result.value<<std::endl;
        std::cout<<"Best args as below:" << std::endl;
        for (int i = 0; i < _gSearchArgs.dimension; i++)
        {
            std::cout<<"\tArgs["<< i<<"] = "<< _gbPosition[i]<<std::endl;
        }
        std::cout<<std::endl;

        return result;
    }

    void snapShot()
    {
        for (int i = 0; i < _predatorNum; i++)
        {
            std::cout<<"\n##Predator "<<i<<"\n";
            _predators[i].printArgs();
        }
        
    }
};