package scalarank

import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.weights.WeightInit

/**
  * Utility functions that may be shared across classes
  */
package object util {

  def arrayToDenseNet(builder: ListBuilder,
                      input: Int,
                      hidden: Array[Int],
                      weightInit: WeightInit = WeightInit.XAVIER): (Int, ListBuilder) = {
    var build = builder
    var in: Int = input
    for (h <- hidden.indices) {
      build = build.layer(h, new DenseLayer.Builder()
        .nIn(in)
        .nOut(hidden(h))
        .activation("relu")
        .weightInit(weightInit)
        .build())
      in = hidden(h)
    }
    (in, build)
  }

}
