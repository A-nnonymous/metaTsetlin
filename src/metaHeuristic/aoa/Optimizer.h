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

#include "pcg_random.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <climits>
#include <thread>
#include <cmath>
#include <mutex>
using std::vector;

template<typename output, typename funcArgs, typename rangeDtype>
class Optimizer{
public:
    struct shared
    {
        double              &MOP;
        double              &w;
        double              &gbestValue;
        double              &gworstValue;
        vector<rangeDtype>  &gbestPosition;
        vector<rangeDtype>  &gworstPosition;
        shared( double              &mop,
                double              &win,
                double              &gbestv,
                double              &gworstv,
                vector<rangeDtype>  &gbp,
                vector<rangeDtype>  &gwp):
                MOP(mop),gbestValue(gbestv),gworstValue(gworstv),
                gbestPosition(gbp),gworstPosition(gwp),w(win){}
    };

    struct limit
    {
        vector<rangeDtype>  min;
        vector<rangeDtype>  max;
        limit(){}
        limit(vector<rangeDtype> minIn, vector<rangeDtype> maxIn)
        {
            min = minIn;
            max = maxIn;
        }
        limit& operator=(limit other)
        {
            min = other.min;
            max = other.max;
            return *this;
        }
    };
    struct args
    {
        int         dimensionNum;
        limit       lim;
        output      (*evaluateFunc)(funcArgs&);
        funcArgs    gFuncArgs;
    };
    

private:
    static inline const double      k = 1.5;
    static inline const double      Mu = 0.5;

    pcg64_fast                      _rng;
    const limit                     _lim;
    const int                       _dimensionNum;
    double                          &_MOP;
    double                          &_w;
    double                          &_gbestValue;
    double                          &_gworstValue;
    vector<rangeDtype>              &_gbestPosition;
    vector<rangeDtype>              &_gworstPosition;

    vector<rangeDtype>              _myPosition;
    funcArgs                        _myFuncArgs;
    output                          _myProperty;
    output                          (*_evaluateFunc)(funcArgs&);

    void rangeRestrict(vector<rangeDtype> &original)
    {
        for (int dim = 0; dim < _dimensionNum; dim++)
        {
            original[dim] = original[dim] > _lim.min[dim]?
                            original[dim] : _lim.min[dim];
            original[dim] = original[dim] < _lim.max[dim]?
                            original[dim] : _lim.max[dim];
        }        
    }

public:
    Optimizer(args arg, shared shr):
    _lim(arg.lim),
    _dimensionNum(arg.dimensionNum),
    _MOP(shr.MOP),
    _w(shr.w),
    _gbestValue(shr.gbestValue),
    _gbestPosition(shr.gbestPosition),
    _gworstValue(shr.gworstValue),
    _gworstPosition(shr.gworstPosition)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg64_fast _rng(seed_source);
        _evaluateFunc = arg.evaluateFunc;
        _myProperty = output();
        _myFuncArgs = arg.gFuncArgs;
        _myPosition.resize(arg.dimensionNum,0);
        for (int i = 0; i < _dimensionNum; i++)
        {
            rangeDtype thisMin = _lim.min[i];
            rangeDtype thisMax = _lim.max[i];
            std::uniform_real_distribution<double> dist(thisMin, thisMax);
            _myPosition[i] = (rangeDtype) dist(_rng);
        }
    }

    void exploit()
    {
        _myFuncArgs.vars = _myPosition;
        _myProperty = _evaluateFunc(_myFuncArgs);
    }

    void explore()
    {
        static std::uniform_real_distribution<double>  dist(0, 1);
        vector<rangeDtype> proposed(_dimensionNum,0);
        double MOA = 1- pow((_gworstValue - _myProperty.value) / (_gbestValue - _gworstValue),k);

        for (int dim = 0; dim < _dimensionNum; dim++)
        {
            double r1 = dist(_rng);
            double r2 = dist(_rng);
            double r3 = dist(_rng);
            if(r1 > MOA)
            {
                if(r2 < 0.5)
                {
                    proposed[dim] = _gbestPosition[dim] / (_MOP + __DBL_MIN__)
                                    *((_lim.max[dim] - _lim.min[dim]) * Mu + _lim.min[dim]);
                }
                else
                {
                    proposed[dim] = _gbestPosition[dim] * _MOP
                                    *((_lim.max[dim] - _lim.min[dim]) * Mu + _lim.min[dim]);
                }
            }
            else
            {
                if(r3 < 0.5)
                {
                    proposed[dim] = _gbestPosition[dim] - _MOP
                                    *((_lim.max[dim] - _lim.min[dim]) * Mu + _lim.min[dim]); 
                    /*
                    proposed[dim] = _w * _myPosition[dim] + 
                                    _MOP * sin(2 * M_PI * dist(_rng)) *
                                    fabs(2 * dist(_rng) * _gbestPosition[dim] - _myPosition[dim]);
                    */
                }
                else
                {
                    proposed[dim] = _gbestPosition[dim] + _MOP
                                    *((_lim.max[dim] - _lim.min[dim]) * Mu + _lim.min[dim]); 
                    /*
                    proposed[dim] = _w * _myPosition[dim] + 
                                    _MOP * cos(2 * M_PI * dist(_rng)) *
                                    fabs(2 * dist(_rng) * _gbestPosition[dim] - _myPosition[dim]);
                    */
                }
            }
        }
        rangeRestrict(proposed);
        _myPosition = proposed;
    }

    double getValue()
    {
        return _myProperty.value;
    }
    vector<rangeDtype> getPosition()
    {
        return _myPosition;
    }
    output getProperty()
    {
        return _myProperty;
    }
};