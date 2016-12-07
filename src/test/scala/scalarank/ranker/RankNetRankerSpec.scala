package scalarank.ranker

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.FlatSpec

import scalarank.datapoint.SVMRankDatapoint
import scalarank.{TestData, metrics}

/**
  * Test specification for the Linear Regression ranker
  */
class RankNetRankerSpec extends FlatSpec {

  "A RankNet loss function" should "be close to 0 when correctly predicted" in {

    // For x_i output 5.0 and set its label as 5.0
    val s_i = Nd4j.create(Array(5.0))
    val y_i = 5.0

    // Create loss
    val loss = new RankNetLoss()
    loss.s_i = s_i
    loss.y_i = y_i

    // For x_j values predict 0.0 and set outputs to 0.0
    val labels = Nd4j.create(Array(0.0, 0.0, 0.0))
    val outputs = Nd4j.create(Array(0.0, 0.0, 0.0))

    // Compute cost
    val cost = loss.computeScore(labels, outputs, "identity", null, true)
    assert(cost < 0.01)

  }

  it should "have a gradient ∇ for which: lim h→0 (‖f(x+h) - f(x) - ∇f(x) · h‖ / ‖h‖) ≈ 0" in {

    // For x_i output 5.0 and set its label as 5.0
    val s_i = Nd4j.create(Array(5.0))
    val y_i = 5.0

    // Create loss
    val loss = new RankNetLoss()
    loss.s_i = s_i
    loss.y_i = y_i

    // Set up labels and x sample data
    val labels = Nd4j.create(Array(0.0, 1.0, 0.0, 4.0))
    val x = Nd4j.create(Array(0.1, -2.0, 7.0, 3.4))

    // Set up gradient check
    val ε = 1e-5
    val grad = loss.computeGradient(labels, x, "identity", null) + 1 // + 1 for identity activation function
    def f(x: INDArray): INDArray = loss.computeScoreArray(labels, x, "identity", null)
    val h = Nd4j.create(Array(1.0, -1.0, 1.3, -2.0)) * ε

    // Check limit approximate to 0
    val lim = Nd4j.norm2(f(x+h) - f(x) - grad * h) / Nd4j.norm2(h)
    assert(lim < 0.01)

  }

}
