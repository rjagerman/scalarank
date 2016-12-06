package scalarank.datapoint

/**
  * A query
  *
  * @param id The query identifier
  * @param datapoints An array of data points representing query-document pairs
  */
class Query[A <: Datapoint](val id: Int, val datapoints: IndexedSeq[A])
