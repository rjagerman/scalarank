package scalarank.ranker

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * An abstract ranker interface
  *
  * @tparam TrainType Type to train on which needs to be at least Datapoint with Relevance
  * @tparam RankType Type to rank with which needs to be at least Datapoint
  */
trait Ranker[TrainType <: Datapoint with Relevance, RankType <: Datapoint] {

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  def train(data: Array[Query[TrainType]]): Unit

  /**
    * Ranks given set of data points
    *
    * @param data The set of data points
    * @return An ordered list of data points
    */
  def rank(data: Array[RankType]): Array[RankType]

  /**
    * Sorts given data using given set of scores
    *
    * @param data The data
    * @param scores The computed scores
    * @return A sorted array of ranks
    */
  protected def sort[R <: RankType : ClassTag](data: Array[R], scores: Array[Double]): Array[R] = {
    data.zip(scores).sortBy(_._2).map(_._1)
  }

}

