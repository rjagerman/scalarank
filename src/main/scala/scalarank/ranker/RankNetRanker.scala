package scalarank.ranker

import org.apache.commons.math3.util.Pair
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.lossfunctions.{ILossFunction, LossFunctions}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}
import scalarank.util.arrayToDenseNet

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
                                                                                            val iterations: Int = 10,
                                                                                            val learningRate: Double = 5e-5)
  extends Ranker[TrainType, RankType] {

  private val loss = new RankNetLoss(σ)

  // Construct deep network
  val config = {
    val listBuilder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(1)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.ADAM)
      .list()
    val (out, net) = arrayToDenseNet(listBuilder, features, hidden)
    net.layer(hidden.length, new OutputLayer.Builder(loss)
        .activation("identity")
        .nIn(out)
        .nOut(1)
        .build())
        .pretrain(false).backprop(true).build()
  }

  val network = new MultiLayerNetwork(config)

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  override def train(data: Iterator[Query[TrainType]]): Unit = {

    for(t <- 0 until iterations) {
      data.foreach { query =>
        val datapoints = query.datapoints

        // Iterate over datapoints in this query
        for (i <- datapoints.indices) {

          // Keep data point x_i fixed
          val x_i = datapoints(i).features
          val y_i = datapoints(i).relevance
          val s_i = network.output(x_i)
          loss.y_i = y_i
          loss.s_i = s_i

          // Train on all data points excluding x_i
          val otherDatapoints = datapoints.zipWithIndex.filter(_._2 != i).map(_._1)
          val X = toMatrix[TrainType](otherDatapoints)
          val y = otherDatapoints.map(_.relevance).toNDArray
          network.fit(X, y)
        }
      }
    }

  }

  /**
    * Ranks given set of data points
    *
    * @param data The set of data points
    * @return An ordered list of data points
    */
  override def rank(data: Array[RankType]): Array[RankType] = {
    val X = toMatrix(data)
    val y = network.output(X)
    val scores = (0 until y.length()).toArray.map(i => y(i))
    sort(data, scores)
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

  /**
    * Score of the current (pairwise) comparison sample x_i
    */
  var s_i: INDArray = Nd4j.zeros(1)

  /**
    * Label of the current (pairwise) comparison sample x_i
    */
  var y_i: Double = 0.0

  override def computeGradientAndScore(labels: INDArray,
                                       preOutput: INDArray,
                                       activationFn: String,
                                       mask: INDArray,
                                       average: Boolean): Pair[java.lang.Double, INDArray] = {
    val s_j = output(preOutput, activationFn)
    val S_ij = Sij(labels)
    Pair.create(score(scoreArray(s_j, S_ij), average), gradient(s_j, S_ij))
  }

  override def computeGradient(labels: INDArray,
                               preOutput: INDArray,
                               activationFn: String,
                               mask: INDArray): INDArray = {
    gradient(output(preOutput, activationFn), Sij(labels))
  }

  override def computeScoreArray(labels: INDArray,
                                 preOutput: INDArray,
                                 activationFn: String,
                                 mask: INDArray): INDArray = {
    scoreArray(output(preOutput, activationFn), Sij(labels))
  }

  override def computeScore(labels: INDArray,
                            preOutput: INDArray,
                            activationFn: java.lang.String,
                            mask: INDArray,
                            average: Boolean): Double = {
    score(scoreArray(output(preOutput, activationFn), Sij(labels)), average)
  }

  /**
    * Computes dC / ds_j, the derivative with respect to s_j, the network's outputs
    *
    * @param s_j The outputs of the network
    * @param S_ij The pairwise labels
    * @return The derivative
    */
  private def gradient(s_j: INDArray, S_ij: INDArray): INDArray = {
    -(-sigmoid((-s_j + s_i) * -σ) + (-S_ij + 1) * 0.5) * σ
  }

  /**
    * Computes the score as an average or sum
    *
    * @param scoreArray The array of scores
    * @param average Whether to average or not
    * @return The cost as a single numerical score
    */
  private def score(scoreArray: INDArray, average: Boolean): Double = average match {
    case true => Nd4j.mean(scoreArray)(0)
    case false => Nd4j.sum(scoreArray)(0)
  }

  /**
    * Computes the score array
    *
    * @param s_j The output of the network for every j
    * @param S_ij The label comparison S_ij
    * @return The cost function array per sample
    */
  private def scoreArray(s_j: INDArray, S_ij: INDArray): INDArray = {
    ((-S_ij - 1) * 0.5 * (-s_j + s_i) * σ) + log(exp((-s_j + s_i) * -σ) + 1)
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

  /**
    * Computes S_ij = {
    *    1.0  if y_i < y_j
    *    0.0  if y_i = y_j
    *   -1.0  if y_i > y_j
    * }
    *
    * @param labels The labels y_j
    * @return Array with values in {0, -1.0, 1.0}
    */
  private def Sij(labels: INDArray): INDArray = labels.gt(y_i) - labels.lt(y_i)

}

