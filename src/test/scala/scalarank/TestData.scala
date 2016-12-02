package scalarank

import org.nd4j.linalg.api.ndarray.INDArray

import scalarank.datapoint.{Datapoint, Query, Relevance, SVMRankDatapoint}

/**
  * An object that contains test data
  */
object TestData {

  val featureless: Array[Datapoint with Relevance] = Array(
    4.0, 3.0, 4.0, 3.0, 1.0, 2.0, 1.0, 4.0, 0.0, 4.0, 0.0, 2.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0
  ).map(r => new FeaturelessDatapointRelevance(r))

  val featurelessPrecision = 0.7272727272727273
  val featurelessAveragePrecision = 0.9343461583351291
  val featurelessReciprocalRank = 1.000000
  val featurelessDCG = 16.31221516353917
  val featurelessnDCG = 0.937572811083981

  val sampleTrainData = readSVMRank("/train.txt")
  val sampleTestData = readSVMRank("/test.txt")

  def readSVMRank(file: String): Array[Query[SVMRankDatapoint]] = {
    val samples = scala.io.Source.fromInputStream(getClass.getResourceAsStream(file)).
      getLines.map(l => SVMRankDatapoint(l))
    samples.toArray.groupBy(d => d.qid).map { case (qid, ds) =>
      new Query[SVMRankDatapoint](qid, ds)
    }.toArray
  }

  /**
    * A datapoint with relevance that does not contain features
    *
    * @param r The relevance label
    */
  class FeaturelessDatapointRelevance(r: Double) extends Datapoint with Relevance {
    override val features: INDArray = null
    override val relevance: Double = r
  }

}
