package scalarank.ranker

import org.scalatest.FlatSpec
import scalarank.TestData
import scalarank.datapoint.{Datapoint, Relevance}

/**
  * Test specification for the Oracle ranker
  */
class OracleRankerSpec extends FlatSpec {

  "An Oracle ranker" should "rank perfectly on our test data" in {
    val oracle = new OracleRanker[Datapoint with Relevance]
    val data = TestData.featureless
    val ranking = oracle.rank(data)
    assert((ranking, ranking.drop(1)).zipped.forall { case (x,y) => x.relevance >= y.relevance })
  }

}
