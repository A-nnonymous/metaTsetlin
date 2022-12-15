#pragma once
#include "Particle.h"

using std::vector;
using std::thread;


template<typename output, typename funcArgs, typename rangeDtype>
class PSOoptimizer{
public:
    struct args
    {
        output                  (*evaluateFunc)(funcArgs&);
        funcArgs                gFuncArgs;
        int                     particleNum;
        int                     dimension;
        int                     maxIter;
        double                  omega;
        double                  ego;
        double                  dt;
        vector<rangeDtype>      rangeMin,rangeMax;
        vector<rangeDtype>      vMin,vMax;
    };

private:
    const args                 _args;
    vector<rangeDtype>          _gBestPosition;
    output                      _gBestProperty;
    vector<Particle
            <
            output,
            funcArgs,
            rangeDtype
            >>                   _particles;

public:
    PSOoptimizer(args args)
    :
    _args(args)
    {
        _gBestPosition.resize(args.dimension,0);
        _gBestProperty = output();

        typename Particle<output,funcArgs,rangeDtype>::particleLimits lim;
        lim.rangeLowerBound = args.rangeMin;
        lim.rangeUpperBound = args.rangeMax;
        lim.velocityLowerBound = args.vMin;
        lim.velocityUpperBound = args.vMax;
        typename Particle<output,funcArgs,rangeDtype>::searchArgs pArgs;
        pArgs.dimension = args.dimension;
        pArgs.maxIter = args.maxIter;
        pArgs.omega = args.omega;
        pArgs.ego = args.ego;
        pArgs.dt = args.dt;
        pArgs.evaluateFunc = args.evaluateFunc;
        pArgs.gFuncArgs = args.gFuncArgs;
        pArgs.limits = lim;
        for (int i = 0; i < args.particleNum; i++)
        {
            _particles.push_back(Particle<output,funcArgs,rangeDtype>(i,pArgs,_gBestPosition,_gBestProperty));
        }
    }
    output optimize()
    {
        std::cout<< "\n############## PSO optimizer started #################"<<std::endl;
        vector<thread> threadPool;
        for (int i = 0; i < _args.particleNum; i++)
        {
            thread th(&Particle<output,funcArgs,rangeDtype>::run, &_particles[i], _args.maxIter);
            threadPool.push_back(std::move(th));
        }
        for (int i = 0; i < _args.particleNum; i++)
        {
            threadPool[i].join();
        }

        std::cout<<"\nOptimization completed "<< ", best value is optimized to:" <<_gBestProperty.value<<std::endl;
        std::cout<<"Best args as below:" << std::endl;
        for (int i = 0; i < _args.dimension; i++)
        {
            std::cout<<"\tArgs["<< i<<"] = "<< _gBestPosition[i] <<std::endl;
        }
        std::cout<<std::endl;
        auto result = _gBestProperty;
        return result;
    }
};
