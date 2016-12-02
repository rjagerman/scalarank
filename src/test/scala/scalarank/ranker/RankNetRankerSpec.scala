package scalarank.ranker

import org.scalatest.FlatSpec

import scalarank.datapoint.SVMRankDatapoint
import scalarank.{TestData, metrics}

/**
  * Test specification for the Linear Regression ranker
  */
class RankNetRankerSpec extends FlatSpec {

  "A RankNet ranker" should "perform better than random" in {
    val trainData = TestData.sampleTrainData
    val testData = TestData.sampleTestData

    val featureSize = trainData(0).datapoints(0).features.length()
    val rankNetRanker = new RankNetRanker[SVMRankDatapoint, SVMRankDatapoint](featureSize)
    rankNetRanker.train(trainData.toIterator)

    // Measure metrics for K@10
    val randomRankings = testData.map(d => d.datapoints)
    val ranknetRankings = testData.map(d => rankNetRanker.rank(d.datapoints))

    // K = 1, 3, 10
    Array(1, 3, 10).foreach { K =>

      // nDCG@K
      val randomNdcgAtK = metrics.meanAtK(randomRankings, metrics.ndcg[SVMRankDatapoint], K)
      val ranknetNdcgAtK = metrics.meanAtK(ranknetRankings, metrics.ndcg[SVMRankDatapoint], K)
      info(s"Random nDCG@$K: ${randomNdcgAtK.formatted("%.4f")}    RankNet nDCG@$K: ${ranknetNdcgAtK.formatted("%.4f")}")
      assert(ranknetNdcgAtK > randomNdcgAtK)

      // MAP@K
      val randomMapAtK = metrics.meanAtK(randomRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
      val ranknetMapAtK = metrics.meanAtK(ranknetRankings, metrics.averagePrecision[SVMRankDatapoint], 10)
      info(s"Random MAP@$K:  ${randomMapAtK.formatted("%.4f")}    RankNet MAP@$K:  ${ranknetMapAtK.formatted("%.4f")}")
      assert(ranknetMapAtK > randomMapAtK)
    }

    // nDCG
    val randomNdcg = metrics.mean(randomRankings, metrics.ndcg[SVMRankDatapoint])
    val ranknetNdcg = metrics.mean(ranknetRankings, metrics.ndcg[SVMRankDatapoint])
    info(s"Random nDCG: ${randomNdcg.formatted("%.4f")}    RankNet nDCG: ${ranknetNdcg.formatted("%.4f")}")
    assert(ranknetNdcg > randomNdcg)

    // MAP@10
    val randomMap = metrics.mean(randomRankings, metrics.averagePrecision[SVMRankDatapoint])
    val ranknetMap = metrics.mean(ranknetRankings, metrics.averagePrecision[SVMRankDatapoint])
    info(s"Random MAP:  ${randomMap.formatted("%.4f")}    RankNet MAP:  ${ranknetMap.formatted("%.4f")}")
    assert(ranknetMap > randomMap)
  }

}
