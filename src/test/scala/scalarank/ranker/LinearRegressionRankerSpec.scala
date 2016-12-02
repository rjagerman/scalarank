package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.{TestData, metrics}
import scalarank.datapoint.SVMRankDatapoint

/**
  * Test specification for the Linear Regression ranker
  */
class LinearRegressionRankerSpec extends FlatSpec {

  "A Linear Regression ranker" should "perform better than random" in {
    val trainData = TestData.sampleTrainData
    val testData = TestData.sampleTestData

    val featureSize = trainData(0).datapoints(0).features.length()
    val linearRegressionRanker = new LinearRegressionRanker[SVMRankDatapoint, SVMRankDatapoint](featureSize)
    linearRegressionRanker.train(trainData.toIterator)

    // Measure metrics for K@10
    val randomRankings = testData.map(d => d.datapoints)
    val linregRankings = testData.map(d => linearRegressionRanker.rank(d.datapoints))

    // K = 3, 5, 10
    Array(3, 5, 10).foreach { K =>

      // nDCG@K
      val randomNdcgAtK = metrics.meanAtK(randomRankings, metrics.ndcg[SVMRankDatapoint], K)
      val linregNdcgAtK = metrics.meanAtK(linregRankings, metrics.ndcg[SVMRankDatapoint], K)
      info(s"Random nDCG@$K: ${randomNdcgAtK.formatted("%.4f")}    Linreg nDCG@$K: ${linregNdcgAtK.formatted("%.4f")}")
      assert(linregNdcgAtK > randomNdcgAtK)

      // MAP@K
      val randomMapAtK = metrics.meanAtK(randomRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
      val linregMapAtK = metrics.meanAtK(linregRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
      info(s"Random MAP@$K:  ${randomMapAtK.formatted("%.4f")}    Linreg MAP@$K:  ${linregMapAtK.formatted("%.4f")}")
      assert(linregMapAtK > randomMapAtK)
    }

    // nDCG
    val randomNdcg = metrics.mean(randomRankings, metrics.ndcg[SVMRankDatapoint])
    val linregNdcg = metrics.mean(linregRankings, metrics.ndcg[SVMRankDatapoint])
    info(s"Random nDCG: ${randomNdcg.formatted("%.4f")}    Linreg nDCG: ${linregNdcg.formatted("%.4f")}")
    assert(linregNdcg > randomNdcg)

    // MAP@10
    val randomMap = metrics.mean(randomRankings, metrics.averagePrecision[SVMRankDatapoint])
    val linregMap = metrics.mean(linregRankings, metrics.averagePrecision[SVMRankDatapoint])
    info(s"Random MAP:  ${randomMap.formatted("%.4f")}    Linreg MAP:  ${linregMap.formatted("%.4f")}")
    assert(linregMap > randomMap)
  }

}
