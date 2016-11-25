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
    val randomRankings = data.map(d => d.datapoints.take(K))
    val linregRankings = data.map(d => linearRegressionRanker.rank(d.datapoints).take(K))

    // Ndcg
    val randomNdcg = metrics.mean(randomRankings, metrics.ndcg[SVMRankDatapoint])
    val linregNdcg = metrics.mean(linregRankings, metrics.ndcg[SVMRankDatapoint])
    assert(linregNdcg > randomNdcg)

    // MAP
    val randomMap = metrics.mean(randomRankings, metrics.averagePrecision[SVMRankDatapoint])
    val linregMap = metrics.mean(linregRankings, metrics.averagePrecision[SVMRankDatapoint])
    assert(linregMap > randomMap)
  }

}
