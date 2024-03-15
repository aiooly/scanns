/**
 * Copyright 2018 LinkedIn Corporation. All rights reserved. Licensed under the BSD-2 Clause license.
 * See LICENSE in the project root for license information.
 */
package com.linkedin.nn

import com.linkedin.nn.algorithm.{CosineSignRandomProjectionNNS, JaccardMinHashNNS}
import com.linkedin.nn.lsh.SignRandomProjectionHashFunction
import com.linkedin.nn.test.SparkTestUtils
import com.linkedin.nn.utils.TopNQueue
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.testng.annotations.Test

/* A simple end-to-end run of a nearest neighbor search using the driver */
class ModelTest extends SparkTestUtils {
  // This is unable to be run currently due to some PriorityQueue serialization issues
  @Test
  def testModel(): Unit = sparkTest("modelTest") {
    sc.getConf.registerKryoClasses(Array(classOf[TopNQueue], classOf[SignRandomProjectionHashFunction]))

    val file = "src/test/resources/nn/example.tsv"
    val data = sc.textFile(file)
      .map { line =>
        val split = line.split(" ")
        (split.head, Vectors.dense(split.tail.map(_.toDouble)))
      }
      .zipWithIndex
    val words = data.map { case (x, y) => (y, x._1) }
    val items = data.map { case (x, y) => (y, x._2) }
    words.cache()
    items.cache()
    val numFeatures = items.values.take(1)(0).size

    val model = new CosineSignRandomProjectionNNS()
      .setNumHashes(200)
      .setSignatureLength(10)
      .setBucketLimit(10)
      .setJoinParallelism(5)
      .createModel(numFeatures)
    val nbrs = model.getSelfAllNearestNeighbors(items, 10)

    print("nbrs: ", nbrs.take(10).mkString("Array(", ", ", ")"))
  }


  @Test
  def testMinHash(): Unit = sparkTest("modelTest") {
    sc.getConf.registerKryoClasses(Array(classOf[TopNQueue], classOf[SignRandomProjectionHashFunction]))

    var df = sparkSession.read.option("header", true).csv("src/test/resources/text01.csv")

    val tokenizer = new Tokenizer().setInputCol("question_text").setOutputCol("words")
    val tokenized = tokenizer.transform(df)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("vectors")
      .setVectorSize(3)
      .setMinCount(0)
    val word2VecModel = word2Vec.fit(tokenized)
    val result = word2VecModel.transform(tokenized)

    val items = result.select("id", "vectors").rdd.map(row => (row.getString(0).toLong, row.getAs[Vector](1)))

    val numFeatures = items.values.take(1)(0).size

//    val model = new JaccardMinHashNNS()
//      .setNumHashes(200)
//      .setSignatureLength(10)
//      .setBucketLimit(10)
//      .setJoinParallelism(5)
//      .createModel(numFeatures)

    val model = new CosineSignRandomProjectionNNS()
      .setNumHashes(200)
      .setSignatureLength(10)
      .setBucketLimit(10)
      .setJoinParallelism(5)
      .createModel(numFeatures)

    val nbrs = model.getSelfAllNearestNeighbors(items, 10)

    nbrs.collect().foreach(row => println(row))

  }
}
