package scalarank.ranker

import scalarank.metrics._

/**
  * Test specification for the Random ranker
  */
class RandomRankerSpec extends RankerSpec {

  "A random ranker" should "report appropriate nDCG results on MQ2008 Fold 1" in {
    testRanker(new RandomRanker(42), ndcg, "nDCG")
  }

}
