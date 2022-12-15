#include "pcg_random.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <climits>
#include <thread>
#include <mutex>

using std::vector;
using std::mutex;

template<typename output, typename funcArgs, typename rangeDtype>
class Particle{
public:
    struct particleLimits
    {
        vector<rangeDtype>     rangeLowerBound, rangeUpperBound;
        vector<rangeDtype>     velocityLowerBound, velocityUpperBound;
        particleLimits(){};
        particleLimits& operator=(particleLimits other)
        {
            rangeLowerBound = other.rangeLowerBound;
            rangeUpperBound = other.rangeUpperBound;
            velocityLowerBound = other.velocityLowerBound;
            velocityUpperBound = other.velocityUpperBound;
            return *this;
        }
    };

    struct searchArgs
    {   
        int                     dimension;
        int                     maxIter;
        double                  omega;
        double                  ego;
        double                  dt;
        output                  (*evaluateFunc)(funcArgs&);
        funcArgs                gFuncArgs;
        particleLimits          limits;
        searchArgs(){}
        searchArgs& operator=(searchArgs other)
        {
            dimension = other.dimension;
            maxIter = other.maxIter;
            omega = other.omega;
            ego = other.ego;
            dt = other.dt;
            evaluateFunc = other.evaluateFunc;
            gFuncArgs = other.gFuncArgs;
            limits = other.limits;
            return *this;
        }
    };

private:
    const int                           _id;
    static inline mutex                 _gBestLock;
    vector<rangeDtype>                  &_gBestPosition;
    output                              &_gBestProperty;
    pcg64_fast                          _rng;
    vector<rangeDtype>                  _pBestPosition;
    output                              _pBestProperty;
    searchArgs                          _pSearchArgs;
    funcArgs                            _pFuncArgs;
    output                              (*_evaluateFunc)(funcArgs&);
    output                              _property;

    vector<rangeDtype>                  _velocity;
    vector<rangeDtype>                  _position;

    void velocityRestrict(vector<rangeDtype> &original)
    {
        for (int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            original[dim] = original[dim] > _pSearchArgs.limits.velocityLowerBound[dim]?
                            original[dim] : _pSearchArgs.limits.velocityLowerBound[dim];
            original[dim] = original[dim] < _pSearchArgs.limits.velocityUpperBound[dim]?
                            original[dim] : _pSearchArgs.limits.velocityUpperBound[dim];
        }
    }
    void rangeRestrict(vector<rangeDtype> &original)
    {
        for (int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            original[dim] = original[dim] > _pSearchArgs.limits.rangeLowerBound[dim]?
                            original[dim] : _pSearchArgs.limits.rangeLowerBound[dim];
            original[dim] = original[dim] < _pSearchArgs.limits.rangeUpperBound[dim]?
                            original[dim] : _pSearchArgs.limits.rangeUpperBound[dim];
        }
    }

    void exploit()
    {
        _pFuncArgs.vars = _position;
        _property = _evaluateFunc(_pFuncArgs);
    }

    void contribute()
    {
        do
        {
            if(_property.value <= _gBestProperty.value)return;   // If not valid competitor, exit and perform explore.
        } 
        while(!_gBestLock.try_lock());                        // Compete for mutex.
        /////////////////////////Critical Section//////////////////////////////////
        if(_property.value > _gBestProperty.value)
        {
            _gBestPosition = _position;
            _gBestProperty = _property;
            _gBestLock.unlock();
        }
        else
        {
            _gBestLock.unlock();
        }
        /////////////////////////Critical Section//////////////////////////////////
    }
    void explore()
    {
        std::uniform_real_distribution<double> dist01(0,1);
        double r1 = dist01(_rng);
        double r2 = dist01(_rng);
        for (int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            _velocity[dim] = rangeDtype(_pSearchArgs.omega * _velocity[dim] +
                                        r1 * _pSearchArgs.ego * (_pBestPosition[dim] - _position[dim]) +
                                        r2 / _pSearchArgs.ego * (_gBestPosition[dim] - _position[dim]));
        }
        velocityRestrict(_velocity);
        for (int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            _position[dim] += _velocity[dim] * _pSearchArgs.dt;
        }
        rangeRestrict(_position);
    }

public:
    Particle(   int                     id,
                searchArgs              gSearchArgs,
                vector<rangeDtype>      &gBestPosition,
                output                  &gBestProperty):
    _id(id),
    _gBestPosition(gBestPosition),
    _gBestProperty(gBestProperty)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg64_fast _rng(seed_source);
        _pSearchArgs = gSearchArgs;
        _property = output();
        _evaluateFunc = _pSearchArgs.evaluateFunc;
        _pFuncArgs = _pSearchArgs.gFuncArgs;
        
        _position.resize(_pSearchArgs.dimension, 0);
        _velocity.resize(_pSearchArgs.dimension, 0);
        _pBestPosition.resize(_pSearchArgs.dimension,0);
        for (int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            rangeDtype thisDimMin = _pSearchArgs.limits.rangeLowerBound[dim];
            rangeDtype thisDimMax = _pSearchArgs.limits.rangeUpperBound[dim];
            rangeDtype thisVMin = _pSearchArgs.limits.velocityLowerBound[dim];
            rangeDtype thisVMax = _pSearchArgs.limits.velocityUpperBound[dim];
            std::uniform_real_distribution<double>  dist(thisDimMin, thisDimMax);
            std::uniform_real_distribution<double>  distV(thisVMin, thisVMax);
            _position[dim] = (rangeDtype) dist(_rng);
            _pBestPosition[dim] = _position[dim];   // Now is the best.
            _velocity[dim] = (rangeDtype) distV(_rng);
        }// Random initialize position to acceptable places.
    }

    void run(int iter)
    {
        for (int i = 0; i < iter; i++)
        {
            exploit();
            contribute();
            explore();
        }
        
    }
};