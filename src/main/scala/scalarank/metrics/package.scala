package scalarank

import scala.reflect.ClassTag
import scalarank.datapoint.{Datapoint, Relevance}
import scalarank.ranker.OracleRanker

/**
  * Provides methods for computing common IR metrics
  *
  * ==Overview==
  * Each method takes as input a sorted array of Datapoints with Relevance labels. The sorting of this array is what
  * will be evaluated.
  *
  * {{{
  *   scala> metrics.precision(ranking)
  *   res0: Double = 0.66666666
  * }}}
  *
  * In order to compute any metric at a cutoff point K, just use the take method on the input array:
  *
  * {{{
  *   scala> val K = 10
  *   K: Int = 10
  *   scala> metrics.precision(ranking.take(K))
  *   res1: Double = 0.7
  * }}}
  *
  */
package object metrics {

  /**
    * Computes precision
    *
    * @param ranking The ranked list
    * @return The precision
    */
  def precision[D <: Datapoint with Relevance](ranking: Seq[D]): Double = {
    ranking.count(d => d.relevance > 0.0).toDouble / ranking.length.toDouble
  }

  /**
    * Computes average precision
    *
    * @param ranking The ranked list
    * @return The average precision
    */
  def averagePrecision[D <: Datapoint with Relevance](ranking: Seq[D]): Double = {
    val relevantDocuments = ranking.zipWithIndex.filter { case (d, i) => d.relevance != 0.0 }
    average(relevantDocuments.zipWithIndex.map { case ((d, i), c) =>
      (c + 1.0) / (i + 1.0)
    })
  }

  /**
    * Computes the reciprocal rank
    *
    * @param ranking The ranked list
    * @return The reciprocal rank
    */
  def reciprocalRank[D <: Datapoint with Relevance](ranking: Seq[D]): Double = {
    ranking.indexWhere(d => d.relevance > 0.0) match {
      case -1 => 0.0
      case x => 1.0 / (1 + x).toDouble
    }
  }

  /**
    * Computes the discounted cumulative gain
    *
    * @param ranking The ranked list
    * @return The discounted cumulative gain
    */
  def dcg[D <: Datapoint with Relevance](ranking: Seq[D]): Double = {
    ranking.zipWithIndex.map {
      case (d, 0) => d.relevance
      case (d, i) => d.relevance * (1.0 / (Math.log(2 + i) / Math.log(2.0)))
    }.sum
  }

  /**
    * Computes the normalized discounted cumulative gain
    *
    * @param ranking The ranked list
    * @return The normalized discounted cumulative gain
    */
  def ndcg[D <: Datapoint with Relevance : ClassTag](ranking: Seq[D]): Double = {
    val oracle = new OracleRanker[D]
    dcg(oracle.rank(ranking.toIndexedSeq)) match {
      case 0 => 0.0
      case perfectDcg => dcg(ranking) / perfectDcg
    }
  }

  /**
    * Computes the mean of a metric over a series of rankings
    *
    * @param rankings The rankings (e.g. per query)
    * @param metric The metric to use
    * @return The mean
    */
  def mean[D <: Datapoint with Relevance](rankings: Iterable[Seq[D]], metric: Seq[D] => Double): Double = {
    meanAtK[D](rankings, metric, rankings.map(r => r.size).max)
  }
  
  /**
   * Computes the mean of a metric over a series of rankings and cutting them off at K
   *
   * @param rankings The rankings (e.g. per query)
   * @param metric The metric to use
   * @param K The cutoff point
   * @return The mean
   */
  def meanAtK[D <: Datapoint with Relevance](rankings: Iterable[Seq[D]], metric: Seq[D] => Double, K: Int): Double = {
    average(rankings.map(ranking => metric(ranking.take(K))))
  }

  /**
    * Computes the average of an iterable of numerics
    *
    * @param ts The iterable
    * @param num The numerical type
    * @tparam T The type
    * @return The average
    */
  private def average[T](ts: Iterable[T])(implicit num: Numeric[T]): Double = ts.size match {
    case 0 => 0.0
    case size => num.toDouble(ts.sum) / size.toDouble
  }

}
