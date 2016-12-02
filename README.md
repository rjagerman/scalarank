# ScalaRank

A modern scala library providing efficient implementations of offline learning to rank algorithms. Under the hood we use
[nd4j](http://nd4j.org/) and [deeplearning4j](https://deeplearning4j.org/) for our scientific computing and neural
network needs.

## Algorithms

Included algorithms are:
* **Oracle**: An oracle ranker that predicts perfectly but requires relevance labels during ranking.
* **Linear Regression**: A linear regression ranker that ranks by predicting scalar values. 
* **[RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)**: A
  neural network with a cost function that minimizes number of wrong inversions.
 
The following algorithms are currently in development:
* **[LambdaRank](http://research.microsoft.com/en-us/um/people/cburges/papers/LambdaRank.pdf)**: An extension to
  RankNet that optimizes non-smooth list-wise metrics directly.
* **[LambdaMART](http://research.microsoft.com/en-us/um/people/cburges/tech_reports/MSR-TR-2010-82.pdf)**: Variant of
  LambdaRank that uses boosted regression trees instead of neural networks.

