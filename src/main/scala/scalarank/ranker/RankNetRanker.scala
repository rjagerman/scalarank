package scalarank.ranker

import org.apache.commons.math3.util.Pair
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.lossfunctions.ILossFunction

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * A RankNet ranker that minimizes number of pair-wise inversions
  *
  * Burges, Chris, et al. "Learning to rank using gradient descent."
  * Proceedings of the 22nd international conference on Machine learning. ACM, 2005.
  *
  * @param features The dimensionality of the input features
  * @param σ The shape of the sigmoid
  * @param hidden An array where each value n corresponds to a dense layer of size n in the network
  * @param seed The random seed
  * @param iterations The number of iterations
  * @param learningRate The learning rate
  * @tparam TrainType Type to train on which needs to be at least Datapoint with Relevance
  * @tparam RankType Type to rank with which needs to be at least Datapoint
  */
class RankNetRanker[TrainType <: Datapoint with Relevance,RankType <: Datapoint : ClassTag](val features: Int,
                                                                                            val σ: Double = 1.0,
                                                                                            val hidden: Array[Int] = Array(10),
                                                                                            val seed: Int = 42,
                                                                                            val iterations: Int = 20,
                                                                                            val learningRate: Double = 5e-5)
  extends Ranker[TrainType, RankType] {

  /**
    * Custom RankNet loss function
    */
  private val loss = new RankNetLoss(σ)

  /**
    * Neural network
    */
  val network = new MultiLayerNetwork({

    // Basic neural network settings
    var build = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(1)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.ADAM)
      .list()

    // Construct hidden layers based on array "hidden"
    var in = features
    for (h <- hidden.indices) {
      build = build.layer(h, new DenseLayer.Builder()
        .nIn(in)
        .nOut(hidden(h))
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .build())
      in = hidden(h)
    }

    // Construct output layer with our custom loss function
    build.layer(hidden.length, new OutputLayer.Builder(loss)
      .activation("identity")
      .nIn(in)
      .nOut(1)
      .build())
      .pretrain(false)
      .backprop(true)
      .build()
  })

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  override def train(data: Iterable[Query[TrainType]]): Unit = {

    for (t <- 0 until iterations) {
      data.foreach { query =>
        val X = toMatrix[TrainType](query.datapoints)
        val y = query.datapoints.map(_.relevance).toNDArray
        network.fit(X, y)
      }
    }

  }

  /**
    * Ranks given set of data points
    *
    * @param data The set of data points
    * @return An ordered list of data points
    */
  override def score(data: IndexedSeq[RankType]): IndexedSeq[Double] = {
    val X = toMatrix(data)
    val y = network.output(X)
    (0 until y.length()).map(i => y(i))
  }

  /**
    * Converts given iterable of data points to an ND4J matrix
    *
    * @param data The data points
    * @tparam D The datapoint type
    * @return A matrix of the features
    */
  private def toMatrix[D <: Datapoint](data: Iterable[D]): INDArray = {
    Nd4j.vstack(data.map(x => x.features).asJavaCollection)
  }

}

/**
  * Loss function for RankNet
  *
  * @param σ The shape of the sigmoid
  */
private class RankNetLoss(σ: Double = 1.0) extends ILossFunction {

  override def computeGradientAndScore(labels: INDArray,
                                       preOutput: INDArray,
                                       activationFn: String,
                                       mask: INDArray,
                                       average: Boolean): Pair[java.lang.Double, INDArray] = {
    val S_var = S(labels)
    val sigma_var = sigma(output(preOutput, activationFn))
    Pair.create(score(S_var, sigma_var, average), gradient(S_var, sigma_var))
  }

  override def computeGradient(labels: INDArray,
                               preOutput: INDArray,
                               activationFn: String,
                               mask: INDArray): INDArray = {
    gradient(S(labels), sigma(output(preOutput, activationFn)))
  }

  override def computeScoreArray(labels: INDArray,
                                 preOutput: INDArray,
                                 activationFn: String,
                                 mask: INDArray): INDArray = {
    scoreArray(S(labels), sigma(output(preOutput, activationFn)))
  }

  override def computeScore(labels: INDArray,
                            preOutput: INDArray,
                            activationFn: java.lang.String,
                            mask: INDArray,
                            average: Boolean): Double = {
    score(S(labels), sigma(output(preOutput, activationFn)), average)
  }

  /**
    * Computes the gradient for the full ranking
    *
    * @param S The S_ij matrix, indicating whether certain elements should be ranked higher or lower
    * @param sigma The sigma matrix, indicating how scores relate to each other
    * @return The gradient
    */
  private def gradient(S: INDArray, sigma: INDArray): INDArray = {
    Nd4j.mean(((-S + 1)*0.5 - sigmoid(-sigma)) * σ, 0).transpose
  }

  /**
    * Computes the score for the full ranking
    *
    * @param S The S_ij matrix, indicating whether certain elements should be ranked higher or lower
    * @param sigma The sigma matrix, indicating how scores relate to each other
    * @return The score array
    */
  private def scoreArray(S: INDArray, sigma: INDArray): INDArray = {
    Nd4j.mean((-S + 1) * 0.5 * sigma + log(exp(-sigma) + 1), 0)
  }

  /**
    * Computes an aggregate over the score, with either summing or averaging
    *
    * @param S The S_ij matrix, indicating whether certain elements should be ranked higher or lower
    * @param sigma The sigma matrix, indicating how scores relate to each other
    * @param average Whether to average or sum
    * @return The score as a single value
    */
  private def score(S: INDArray, sigma: INDArray, average: Boolean): Double = average match {
    case true => Nd4j.mean(scoreArray(S, sigma))(0)
    case false => Nd4j.sum(scoreArray(S, sigma))(0)
  }

  /**
    * Computes the matrix S_ij, which indicates wheter certain elements should be ranked higher or lower
    *
    * S_ij = {
    *    1.0   if y_i > y_j
    *    0.0   if y_i = y_j
    *   -1.0   if y_i < y_j
    * }
    *
    * @param labels The labels
    * @return The S_ij matrix
    */
  private def S(labels: INDArray): INDArray = {
    val labelMatrix = labels.transpose.mmul(Nd4j.ones(labels.rows, labels.columns)) - Nd4j.ones(labels.columns, labels.rows).mmul(labels)
    labelMatrix.gt(0) - labelMatrix.lt(0)
  }

  /**
    * Computes the sigma matrix, which indicates how scores relate to each other
    *
    * sigma_ij = σ * (s_i - s_j)
    *
    * @param outputs The signal outputs from the network
    * @return The sigma matrix
    */
  private def sigma(outputs: INDArray): INDArray = {
    (outputs.transpose.mmul(Nd4j.ones(outputs.rows, outputs.columns)) - Nd4j.ones(outputs.columns, outputs.rows).mmul(outputs)) * σ
  }

  /**
    * Compute output with an activation function
    *
    * @param preOutput The output of the network before applying the activation function
    * @param activationFn The activation function
    * @return The output with given activation function
    */
  private def output(preOutput: INDArray, activationFn: String): INDArray = {
    Nd4j.getExecutioner.execAndReturn(Nd4j.getOpFactory.createTransform(activationFn, preOutput.dup))
  }

}

