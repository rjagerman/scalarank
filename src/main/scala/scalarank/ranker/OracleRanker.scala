package scalarank.ranker

import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * Ranks documents perfectly but requires relevance labels to be known during ranking
  *
  * @tparam T Type to train on and rank with which needs to be at least Datapoint with Relevance
  */
class OracleRanker[T <: Datapoint with Relevance : ClassTag] extends Ranker[T, T] {

  /**
    * Trains the ranker on a set of labeled data points
    *
    * @param data The set of labeled data points
    */
  override def train(data: Iterable[Query[T]]): Unit = {  }

  /**
    * Ranks given set of data points
    *
    * @param data The set of data points
    * @return An ordered list of data points
    */
  override def score(data: IndexedSeq[T]): IndexedSeq[Double] = {
    val maximum = data.map(d => d.relevance).max
    data.map(d => maximum - d.relevance)
  }

}
