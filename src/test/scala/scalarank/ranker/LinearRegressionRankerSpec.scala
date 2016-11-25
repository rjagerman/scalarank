package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.{TestData, metrics}
import scalarank.datapoint.SVMRankDatapoint

/**
  * Test specification for the Linear Regression ranker
  */
class LinearRegressionRankerSpec extends FlatSpec {

  "A Linear Regression ranker" should "perform better than random" in {
    val data = TestData.sampleTrainData

    val featureSize = data(0).datapoints(0).features.length()
    val linearRegressionRanker = new LinearRegressionRanker[SVMRankDatapoint, SVMRankDatapoint](featureSize)
    linearRegressionRanker.train(data)

    // Measure metrics for K@10
    val K = 10
    val randomRankings = data.map(d => d.datapoints)
    val linregRankings = data.map(d => linearRegressionRanker.rank(d.datapoints))

    // Ndcg
    val randomNdcg = metrics.meanAtK(randomRankings, metrics.ndcg[SVMRankDatapoint], 10)
    val linregNdcg = metrics.meanAtK(linregRankings, metrics.ndcg[SVMRankDatapoint], 10)
    assert(linregNdcg > randomNdcg)

    // MAP
    val randomMap = metrics.meanAtK(randomRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
    val linregMap = metrics.meanAtK(linregRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
    assert(linregMap > randomMap)
  }

}
