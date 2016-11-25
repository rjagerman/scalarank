package scalarank.ranker

import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * Ranks documents perfectly but requires relevance labels to be known during ranking
  *
  * @tparam T Type to train on and rank with which needs to be at least Datapoint with Relevance
  */
class OracleRanker[T <: Datapoint with Relevance] extends Ranker[T, T] {

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  override def train(data: Array[Query[T]]): Unit = {  }

  /**
    * Ranks given set of data points
    *
    * @param data The set of data points
    * @return An ordered list of data points
    */
  override def rank(data: Array[T]): Array[T] = {
    data.sortBy(d => -d.relevance)
  }

}
