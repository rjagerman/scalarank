package scalarank.ranker

import org.scalatest.FreeSpec

import scala.collection.mutable
import scala.reflect.ClassTag
import scalarank.TestData
import scalarank.metrics.{averagePrecision, mean, meanAtK, ndcg}
import scalarank.datapoint.{Datapoint, Query, Relevance, SVMRankDatapoint}

/**
  * Testing ranker performance
  */
class RankerSpec extends FreeSpec {

  val trainData = TestData.sampleTrainData
  val testData = TestData.sampleTestData
  val featureSize = trainData(0).datapoints(0).features.length()

  "Testing Oracle ranker performance (nDCG)" in {
    testRanker(new OracleRanker(), ndcg, "nDCG")
  }

  "Testing Random ranker performance (nDCG)" in {
    testRanker(new RandomRanker(42), ndcg, "nDCG")
  }

  "Testing Linear Regression ranker performance (nDCG)" in {
    testRanker(new LinearRegressionRanker(featureSize, seed=42), ndcg, "nDCG")
  }

  "Testing RankNet ranker performance (nDCG)" in {
    testRanker(new RankNetRanker(featureSize, seed=42), ndcg, "nDCG")
  }

  /**
    * Tests a ranker by training it on our training set and testing it on our test set
    *
    * @param ranker The ranker to train and evaluate
    * @param metric The metric to score by
    */
  def testRanker(ranker: Ranker[SVMRankDatapoint, SVMRankDatapoint],
                 metric: Seq[SVMRankDatapoint] => Double,
                 metricName: String = ""): Unit = {
    ranker.train(trainData)
    val rankings = testData.map(d => ranker.rank(d.datapoints))
    Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).foreach { k =>
      val result = meanAtK(rankings, metric, k)
      info(s"$metricName@${k.toString.padTo(4, ' ')} = $result")
    }
    info(s"$metricName mean = ${mean(rankings, metric)}")
  }

}
