name := "ScalaRank"

version := "1.0"

scalaVersion := "2.11.8"


libraryDependencies += "org.nd4j" % "nd4j-native" % "0.6.0" classifier "" classifier "macosx-x86_64"

libraryDependencies += "org.nd4j" %% "nd4s" % "0.6.0"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"

classpathTypes += "maven-plugin"


fork in Test := true


