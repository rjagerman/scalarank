package scalarank.ranker

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * A linear regression ranker that ranks by scoring data points as scalar values.
  *
  * Linear regression is implemented as a single-layer neural network with an MSE loss function and an identity
  * activation function.
  *
  * @param features The dimensionality of the input features
  * @param seed The random seed
  * @param iterations The number of iterations
  * @param learningRate The learning rate
  * @tparam TrainType Type to train on which needs to be at least Datapoint with Relevance
  * @tparam RankType Type to rank with which needs to be at least Datapoint
  */
class LinearRegressionRanker[TrainType <: Datapoint with Relevance,RankType <: Datapoint : ClassTag](val features: Int,
                                                                                                     val seed: Int = 42,
                                                                                                     val iterations: Int = 100,
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
  override def train(data: Iterable[Query[TrainType]]): Unit = {

    val datapoints = data.toArray.flatMap(x => x.datapoints)
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
