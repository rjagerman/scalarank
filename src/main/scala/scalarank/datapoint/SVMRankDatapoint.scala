package scalarank.datapoint

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * A datapoint based on SVM rank syntax
  *
  * @param line The line containing this data point
  */
class SVMRankDatapoint(line: String) extends Datapoint with Relevance {

  override val features: INDArray = {
    val (_, values) = SVMRankDatapoint.FEATURE_REGEX.findAllIn(line).
      map(m => m.split(":")).
      map(m => (m(0).toInt, m(1).toDouble)).
      toArray.sorted.unzip
    Nd4j.create(values)
  }

  override val relevance: Double = SVMRankDatapoint.RELEVANCE_REGEX.findFirstIn(line).get.toDouble

  val qid: Int = line match { case SVMRankDatapoint.QID_REGEX(id) => id.toInt }

}

object SVMRankDatapoint {

  private val QID_REGEX = """.*qid:([0-9]+).*""".r
  private val RELEVANCE_REGEX= """^[0-9]+""".r
  private val FEATURE_REGEX = """[0-9]+:[^ ]+""".r

  def apply(line: String): SVMRankDatapoint = new SVMRankDatapoint(line)

}

