#include "Clause.h"
#include <vector>
#include <immintrin.h>

using std::vector;

alignas(64) struct pack512{
    int data[16];
    pack512()
    {
        for (int i = 0; i < 16; i++)
        {
            data[i] = 1;
        }
        
    }
    pack512(int limit)
    {
        if (limit>16)
        {
            pack512();
        }
        
        for (int i = 0; i < 16; i++)
        {
            if (i<limit)
            {
                data[i] = 1;
            }
            else
            {
                data[i] = 0;
            }
            
        }
        
    }
};
int main()
{
    Clause::ClauseArgs cArgs;
    cArgs.no = 0;
    cArgs.isPositiveClause = true;
    cArgs.inputSize = 8;
    cArgs.specificity = 2; 
    Clause debug(cArgs);
    
    int blockNum = cArgs.inputSize/16 + (cArgs.inputSize%16==0? 0:1);
    vector<pack512> twopac(blockNum,pack512(cArgs.inputSize));
    vector<pack512> result(blockNum,pack512(cArgs.inputSize));
    vector<__m512i> testblocks(blockNum,_mm512_setzero_epi32());
    for (int i = 0; i < blockNum; i++)
    {
        testblocks[i] = _mm512_loadu_epi32(&(twopac[i].data));
    }
    int voteres = debug.vote(testblocks);
    std::cout<<"vote result: "<<voteres <<std::endl;
    debug.feedbackTypeI();
    return 0;
}
