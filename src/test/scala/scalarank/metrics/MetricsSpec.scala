package scalarank.metrics

import org.scalatest.FlatSpec
import scalarank.TestData

/**
  * Test specification for metrics
  */
class MetricsSpec extends FlatSpec {

  "Precision" should "be 1.0 for only relevant documents" in {
    val data = Array(1.0, 1.0, 1.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(precision(data) == 1.0)
  }

  it should "be 0.0 for only non-relevant documents" in {
    val data = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(precision(data) == 0.0)
  }

  it should "be 0.5 when half the documents are relevant" in {
    val data = Array(0.0, 1.0, 0.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(precision(data) == 0.5)
  }

  it should "be invariant to changes in ordering" in {
    val data = Array(
      Array(0.0, 1.0, 0.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x)),
      Array(1.0, 1.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x)),
      Array(1.0, 0.0, 0.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x)),
      Array(0.0, 0.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    )
    data.foreach { d =>
      assert(precision(d) == 0.5)
    }
  }

  it should "be %.4f for our test data set".format(TestData.featurelessPrecision) in {
    assert(precision(TestData.featureless) == TestData.featurelessPrecision)
  }

  "AveragePrecision" should "be 1.0 for only relevant documents" in {
    val data = Array(1.0, 1.0, 1.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(averagePrecision(data) == 1.0)
  }

  it should "be 0.0 for only non-relevant documents" in {
    val data = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(averagePrecision(data) == 0.0)
  }

  it should "be 1.0 when exactly the first half of the documents are relevant" in {
    val data = Array(1.0, 1.0, 1.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(averagePrecision(data) == 1.0)
  }

  it should "be %.4f for our test data set".format(TestData.featurelessAveragePrecision) in {
    assert(averagePrecision(TestData.featureless) == TestData.featurelessAveragePrecision)
  }

  "ReciprocalRank" should "be 1.0 for only relevant documents" in {
    val data = Array(1.0, 1.0, 1.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(reciprocalRank(data) == 1.0)
  }

  it should "be 1.0 when only the first document is relevant" in {
    val data = Array(1.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(reciprocalRank(data) == 1.0)
  }

  it should "be 0.0 for only non-relevant documents" in {
    val data = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(reciprocalRank(data) == 0.0)
  }

  it should "be 0.5 when the second document is the first relevant one" in {
    val data = Array(0.0, 1.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(reciprocalRank(data) == 0.5)
  }

  it should "be 0.3333 when the third document is the first relevant one" in {
    val data = Array(0.0, 0.0, 1.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(reciprocalRank(data) == 0.3333333333333333)
  }

  it should "be %.4f for our test data set".format(TestData.featurelessReciprocalRank) in {
    assert(reciprocalRank(TestData.featureless) == TestData.featurelessReciprocalRank)
  }

  "DCG" should "be 2.1309 for three relevant documents" in {
    val data = Array(1.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(dcg(data) == 2.1309297535714573)
  }

  it should "be 0.0 for only non-relevant documents" in {
    val data = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(dcg(data) == 0.0)
  }

  it should "be %.4f for our test data set".format(TestData.featurelessDCG) in {
    assert(dcg(TestData.featureless) == TestData.featurelessDCG)
  }

  "nDCG" should "be 1.0 for only relevant documents" in {
    val data = Array(1.0, 1.0, 1.0, 1.0, 1.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(ndcg(data) == 1.0)
  }

  it should "be 0.0 for only non-relevant documents" in {
    val data = Array(0.0, 0.0, 0.0, 0.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(ndcg(data) == 0.0)
  }

  it should "be 1.0 for a perfectly ranked list" in {
    val data = Array(5.0, 5.0, 4.0, 2.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(ndcg(data) == 1.0)
  }

  it should "be less than 1.0 for a non-perfectly ranked list" in {
    val data = Array(5.0, 4.0, 5.0, 2.0, 0.0).map(x => new TestData.FeaturelessDatapointRelevance(x))
    assert(ndcg(data) < 1.0)
  }

  it should "be %.4f for our test data set".format(TestData.featurelessnDCG) in {
    assert(ndcg(TestData.featureless) == TestData.featurelessnDCG)
  }

}
