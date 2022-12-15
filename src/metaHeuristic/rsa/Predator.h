#include <vector>
#include <iostream>
#include <random>
#include <climits>
#include "pcg_random.hpp"

/// @brief Definition of individual optimizer class.
/// @tparam output Output struct of fitness function, which must contain fitness value named "value" initalized to (double)-Inf.
/// @tparam funcArgs Fitness function arguments(trivial or struct) passed from user.
/// @tparam rangeDtype Data type of search range.
template<typename output, typename funcArgs, typename rangeDtype>
class Predator{
public:
    struct rangeLimits
    {
        std::vector<rangeDtype>     maxLimits;
        std::vector<rangeDtype>     minLimits;
        rangeLimits(){};
        rangeLimits(std::vector<rangeDtype> min, 
                    std::vector<rangeDtype> max)
        {
            minLimits = min;
            maxLimits = max;
        }
    };
    struct searchArgs
    {   
        int                 dimension;
        int                 maxIter;
        double              alpha;
        double              beta;
        rangeLimits         searchRange;
        searchArgs(){};
        searchArgs( int dim, int iter, 
                    double alpha, double beta,
                    rangeLimits limits)
        {
            dimension = dim;
            maxIter = iter;
            this->alpha = alpha;
            this->beta = beta;
            searchRange = limits;
        };
    };

private:
    //std::mt19937                        _rng;
    pcg64_fast                          _rng;
    const int                           _mynumber;
    output                              _property;
    output                              (*_funcPtr)(funcArgs&);
    searchArgs                          _pSearchArgs;
    funcArgs                            _pFuncArgs;
    std::vector<
        std::vector<rangeDtype>>        &_allPosition;

    void restrict(std::vector<rangeDtype> &original)
    {
        for(int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            original[dim] = original[dim] > _pSearchArgs.searchRange.minLimits[dim]?
                            original[dim] : _pSearchArgs.searchRange.minLimits[dim];
            original[dim] = original[dim] < _pSearchArgs.searchRange.maxLimits[dim]?
                            original[dim] : _pSearchArgs.searchRange.maxLimits[dim];
        }
    }

public:
    /// @brief Constructor of individual optimizers in RSA algorithm
    /// @param fitnessFunc Fitness function used to evaluate arguments.
    /// @param gSearchArgs RSA related arguments passed from hibitat.
    /// @param gFuncArgs Fitness function related arguments passed from hibitat.
    /// @param allPosition Reference of position matrix.
    /// @param mynumber Unique tag of this optimizer.
    Predator(   output                                  (*fitnessFunc)(funcArgs&),
                searchArgs                              gSearchArgs,
                funcArgs                                gFuncArgs,
                std::vector<std::vector<rangeDtype>>     &allPosition,
                int                                     mynumber):
    _allPosition(allPosition),
    _mynumber(mynumber)
    {
        //std::mt19937    _rng(std::random_device{}());
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg64_fast _rng(seed_source);
        _property = output();         // Initialize personal best property to default.
        _funcPtr = fitnessFunc;
        _pSearchArgs = gSearchArgs;
        _pFuncArgs = gFuncArgs;
        _pFuncArgs.vars = _allPosition[mynumber]; // Target for this algorithm, changable arguments vector.
    }
    
    /// @brief Evaluation of fitness function, have side effects on local optima and its property.
    void exploit()
    {
        output thisResult = _funcPtr(_pFuncArgs);
        _property = thisResult;
    }

    /// @brief Explore argument space with different strategy according to current stage.
    /// @param gbest Received global optima argument from previous evaluation.
    /// @param evoSense Evolution factor passed from hibitat.
    /// @param iter Current iteration number.
    void explore(std::vector<rangeDtype> gbest, double evoSense, int iter)
    {
        static std::uniform_int_distribution<int>       choice(0,_allPosition.size() - 1);
        static std::uniform_real_distribution<double>   d01(0, 1);
        std::vector<rangeDtype> proposal;
        proposal.resize(_pSearchArgs.dimension, 0);

        for(int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            int thisChoice = choice(_rng);
            double mymean = std::reduce(_allPosition[_mynumber].begin(),
                                        _allPosition[_mynumber].end()) /
                                        (double) _pSearchArgs.dimension;
            double dimRange = double(_pSearchArgs.searchRange.maxLimits[dim] 
                                        -_pSearchArgs.searchRange.minLimits[dim]);

            double R = gbest[dim] - (_allPosition[thisChoice][dim]) / (gbest[dim] + __FLT_MIN__);
            double P = _pSearchArgs.alpha + (_allPosition[_mynumber][dim] - mymean) / 
                    (gbest[dim] * dimRange + __FLT_MIN__);
            double Eta = gbest[dim] * P;

            if(iter <= (_pSearchArgs.maxIter)/4)                // Stage 1
            {
                proposal[dim] = gbest[dim] - Eta * _pSearchArgs.beta - R * d01(_rng);
            }
            else if (   iter <= ((_pSearchArgs.maxIter)/2) &&
                        iter > ((_pSearchArgs.maxIter)/4))      // Stage 2
            {
                proposal[dim] = gbest[dim] * _allPosition[thisChoice][dim] * evoSense * d01(_rng);
            }
            else if (   iter <= (3*(_pSearchArgs.maxIter)/4) &&
                        iter > ((_pSearchArgs.maxIter)/2))      // Stage 3
            {
                proposal[dim] = gbest[dim] * P * d01(_rng);
            }
            else                                                // Stage 4
            {
                proposal[dim] = gbest[dim] - Eta * __FLT_MIN__ - R * d01(_rng);
            }
        }
        restrict(proposal);

        for(int dim = 0; dim < _pSearchArgs.dimension; dim++)
        {
            _allPosition[_mynumber][dim] = proposal[dim];
        }
        _pFuncArgs.vars = _allPosition[_mynumber]; // Renew the changable vector.
    }

    double getValue()
    {
        return _property.value;
    }
    output getProperty()
    {
        return _property;
    }

    void printArgs()
    {
        for (int i = 0; i < _pSearchArgs.dimension; i++)
        {
            std::cout<<"\tArg["<<i<<"] "<<_allPosition[_mynumber][i]<<"\n";
        }
        
    }
};
