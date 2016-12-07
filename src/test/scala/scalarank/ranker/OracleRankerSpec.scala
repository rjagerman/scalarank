package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.{TestData, metrics}
import scalarank.datapoint.{Datapoint, Relevance}

/**
  * Test specification for the Oracle ranker
  */
class OracleRankerSpec extends FlatSpec {

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

}
