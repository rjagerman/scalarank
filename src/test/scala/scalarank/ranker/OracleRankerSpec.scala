package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.{TestData, metrics}
import scalarank.datapoint.{Datapoint, Relevance}
import scalarank.metrics._

/**
  * Test specification for the Oracle ranker
  */
class OracleRankerSpec extends RankerSpec {

  "An Oracle ranker" should "rank perfectly on our test data" in {
    val oracle = new OracleRanker[Datapoint with Relevance]
    val data = TestData.featureless
    oracle.train(Iterable.empty)
    val ranking = oracle.rank(data)
    assert((ranking, ranking.drop(1)).zipped.forall { case (x,y) => x.relevance >= y.relevance })
  }

  it should "have perfect nDCG on our test data" in {
    val oracle = new OracleRanker[Datapoint with Relevance]
    val data = TestData.featureless
    oracle.train(Iterable.empty)
    val ranking = oracle.rank(data)
    assert(metrics.ndcg(ranking) == 1.0)
  }

  it should "report appropriate nDCG results on MQ2008 Fold 1" in {
    testRanker(new OracleRanker(), ndcg, "nDCG")
  }

}
