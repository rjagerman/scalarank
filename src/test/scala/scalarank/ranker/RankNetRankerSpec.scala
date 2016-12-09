package scalarank.ranker

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, Matchers}

import scalarank.datapoint.SVMRankDatapoint
import scalarank.metrics._
import scalarank.{TestData, metrics}

/**
  * Test specification for the Linear Regression ranker
  */
class RankNetRankerSpec extends RankerSpec with GradientCheck with Matchers {

  "A RankNet ranker" should "report appropriate nDCG results on MQ2008 Fold 1" in {
    testRanker(new RankNetRanker(featureSize, seed=42), ndcg, "nDCG")
  }

  "A RankNet loss function" should "be approximately log(2) when correctly predicted" in {

    // Create loss
    val loss = new RankNetLoss()

    // Single correctly predicted value
    val labels = Nd4j.create(Array(0.0, 0.0, 0.0))
    val outputs = Nd4j.create(Array(0.0, 0.0, 0.0))

    // Compute cost
    val cost = loss.computeScore(labels, outputs, "identity", null, true)
    assert(Math.abs(cost - Math.log(2.0)) < 0.0000001)

  }

  it should "succesfully perform the gradient limit check" in {

    // Create loss
    val loss = new RankNetLoss()

    // Set up labels and x sample data
    val labels = Nd4j.create(Array(0.0, 1.0, 0.0, 4.0))
    val x = Nd4j.create(Array(0.1, -2.0, 7.0, 3.4))

    // Check gradient
    val grad = -loss.computeGradient(labels, x, "identity", null)
    def f(x: INDArray): INDArray = loss.computeScoreArray(labels, x, "identity", null)
    val limits = gradientLimits(grad, x, f)
    info(limits.mkString(" > "))
    limits.sliding(2).foreach { case Array(l1, l2) => assert(l1 > l2) }

  }

  it should "succesfully compute both the gradient and cost" in {

    // Create loss
    val loss = new RankNetLoss()

    // Set up labels and x sample data
    val labels = Nd4j.create(Array(0.0, 1.0, 0.0, 4.0))
    val x = Nd4j.create(Array(0.1, -2.0, 7.0, 3.4))

    // Compute the gradient and score
    val gradient = loss.computeGradient(labels, x, "identity", null)
    val score = loss.computeScore(labels, x, "identity", null, average=true)
    val gradientAndScore = loss.computeGradientAndScore(labels, x, "identity", null, average=true)

    // Check computation
    gradientAndScore.getFirst shouldBe score
    gradientAndScore.getSecond shouldBe gradient

  }

}
