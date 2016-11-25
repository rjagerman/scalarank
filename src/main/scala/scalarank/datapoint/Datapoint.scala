package scalarank.datapoint

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * A data point, this is typically a feature vector containing query-document features
  */
abstract class Datapoint {

  /**
    * The features as a dense vector
    */
  val features: INDArray

}

/**
  * For labeling data points with relevance
  */
trait Relevance {

  /**
    * The relevance of the data point. Typically higher means more relevant.
    */
  val relevance: Double

}

