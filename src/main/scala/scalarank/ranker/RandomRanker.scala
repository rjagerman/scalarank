package scalarank.ranker

import scala.reflect.ClassTag
import scala.util.Random
import scalarank.datapoint.{Datapoint, Query, Relevance}

/**
  * Ranks documents randomly
  *
  * @tparam T Type to train on and rank with which needs to be at least Datapoint with Relevance
  */
class RandomRanker[T <: Datapoint with Relevance, R <: Datapoint : ClassTag](seed: Int) extends Ranker[T, R] {

  val rng = new Random(seed)

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
  override def score(data: IndexedSeq[R]): IndexedSeq[Double] = {
    data.indices.map(_ => rng.nextDouble()).toArray
  }

}
