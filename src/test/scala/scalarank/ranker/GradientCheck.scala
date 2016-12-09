package scalarank.ranker

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Test trait for checking gradient functions
  */
trait GradientCheck {

  /**
    * Computes the gradient limit: lim h→0 (‖f(x+h) - f(x) - ∇f(x) · h‖ / ‖h‖)
    *
    * @param gradient The gradient (as a vector)
    * @param x The input to compute said gradient (as a vector)
    * @param function The function over which the gradient is computed
    * @return The limit
    */
  def gradientLimits(gradient: INDArray, x: INDArray, function: INDArray => INDArray): Array[Double] = {
    val rand = Nd4j.randn(x.rows, x.columns)
    Array(1e1, 1, 1e-1, 1e-2).map { ε =>
      (0 until x.columns).map { i =>
        val e = Nd4j.zeros(x.columns)
        e(i) = 1.0
        val approximateGradient = (function(x + e * ε) - function(x - e * ε)) / (2*ε)
        Math.abs(approximateGradient(i) - gradient(i))
      }.sum
    }
  }

}
