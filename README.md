# metaTsetlin
Using meta-heuristic algorithms to optimize Tsetlin Machine, and implementing multigranular and dropout clauses by the way

This project integrated meta-heuristic algorithms like Reptile Search Algorithm(RSA), Arithmetic Optimization Algorithm(AOA) and Particle Swarm Optimization into Hyper-Parameter optimization of Tsetlin Machine.

Further more, I fully re-constructed previous implementation of Tsetlin Machine and vectorize the prediction and learning process using AVX-512F instruction set(Later AVX-2 will be included).

The Tsetlin Machine itself is now featured with dropout clauses and multigranular feedback sensitivity, which reduced the hyper-parameters' search space to 2-D(clause number and T).

Later on I will start to adding multithreading and job manager to Tsetlin Machine's learning&predicting process, and seek to improve the accuracy through ensemble learning.

