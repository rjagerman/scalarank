package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.{TestData, metrics}
import scalarank.datapoint.SVMRankDatapoint
import scalarank.metrics._

/**
  * Test specification for the Linear Regression ranker
  */
class LinearRegressionRankerSpec extends RankerSpec {

  "A LinearRegression Ranker" should "report appropriate nDCG results on MQ2008 Fold 1" in {
    testRanker(new LinearRegressionRanker(featureSize, seed=42), ndcg, "nDCG")
  }

}
