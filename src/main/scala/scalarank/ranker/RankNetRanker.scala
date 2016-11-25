package scalarank.ranker

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * A RankNet ranker that minimizes number of pair-wise inversions
  *
  * @param features The dimensionality of the input features
  * @param seed The random seed
  * @param iterations The number of iterations
  * @param learningRate The learning rate
  * @tparam TrainType Type to train on which needs to be at least Datapoint with Relevance
  * @tparam RankType Type to rank with which needs to be at least Datapoint
  */
class RankNetRanker[TrainType <: Datapoint with Relevance,RankType <: Datapoint : ClassTag](val features: Int,
                                                                                            val hidden: Array[Int] = Array(100),
                                                                                            val seed: Int = 42,
                                                                                            val iterations: Int = 10,
                                                                                            val learningRate: Double = 1e-3)
  extends Ranker[TrainType, RankType] {

  val network = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
    .seed(seed)
    .iterations(iterations)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .updater(Updater.ADAM)
    .list()
    .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation("identity")
      .nIn(features)
      .nOut(1)
      .build())
    .pretrain(false).backprop(true).build()
  )

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  override def train(data: Array[Query[TrainType]]): Unit = {
    val datapoints = data.flatMap(x => x.datapoints)
    val labels = datapoints.map(x => x.relevance)

    val X = toMatrix[TrainType](datapoints)
    val y = labels.toNDArray

    network.fit(X, y)
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
