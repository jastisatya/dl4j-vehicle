package com.jerry.vehicle.data

import java.util

import com.jerry.vehicle.utils.SparkUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{CacheMode, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerMinMaxScaler, NormalizerStandardize}
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConversions._

object DataLoad {
  /**
    * 随机种子
    */
  val seed: Long = 1L

  /**
    * 训练文件名称
    */
  val trainFileName: String = "train_1.csv"

  /**
    * 测试文件名称
    */
  val testFileName: String = "test_1.csv"

  /**
    * 跳过的行数
    */
  val skipLineNumber: Int = 1

  /**
    * 分隔符
    */
  val delimiter: Char = ','

  /**
    * label所在的位置
    */
  val labelIndex: Int = 0

  /**
    * 批处理的数目
    */
  val batchSize: Int = 150

  /**
    * 迭代次数
    */
  val iterations: Int = 1000

  /**
    * 每个worker处理的数目
    */
  val batchSizePerWorker: Int = 32

  /**
    * 平均时间
    */
  val averagingFrequency: Int = 7

  /**
    * the number of mini batches to asynchronously prefetch in the worker
    */
  val workerPrefetchNumBatches: Int = 2

  def main(args: Array[String]): Unit = {
    val reader: CSVRecordReader = new CSVRecordReader(skipLineNumber, delimiter)
    reader.initialize(new FileSplit(new ClassPathResource(trainFileName).getFile()))

    val trainDataSetIt: RecordReaderDataSetIterator = new RecordReaderDataSetIterator(reader, batchSize, 0, 0, true)

    if (trainDataSetIt.hasNext) {
      val trainDataSet: DataSet = trainDataSetIt.next()
      trainDataSet.shuffle()

      /*数据归一化、正则化*/
      val minMaxScaler: DataNormalization = new NormalizerMinMaxScaler()
      minMaxScaler.fit(trainDataSet)
      minMaxScaler.transform(trainDataSet)

      val normalizeStandardize: DataNormalization = new NormalizerStandardize()
      normalizeStandardize.fit(trainDataSet)
      normalizeStandardize.transform(trainDataSet)

      /*第一层个数*/
      val num1Inputs: Int = trainDataSet.getFeatures.length()
      val num1Outputs: Int = 3 * trainDataSet.getFeatures.length()
      /*第二层个数*/
      val num2Inputs: Int = num1Outputs
      val num2Outputs: Int = (0.6 * num2Inputs).toInt
      /*第三层个数*/
      val num3Inputs: Int = num2Outputs
      val num3Outputs: Int = (0.3 * num3Inputs).toInt
      /*第四层个数*/
      val num4Inputs: Int = num3Outputs
      val num4Outputs: Int = 1

      /*配置网络结构*/
      val networkConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .cacheMode(CacheMode.DEVICE)
        .iterations(iterations)
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.XAVIER)
        .learningRate(0.01)
        .regularization(true).l2(1e-4)
        .updater(Updater.NESTEROVS)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list()
        .layer(0, new DenseLayer.Builder().nIn(num1Inputs).nOut(num1Outputs).build())
        .layer(1, new DenseLayer.Builder().nIn(num2Inputs).nOut(num2Outputs).build())
        .layer(2, new DenseLayer.Builder().nIn(num3Inputs).nOut(num3Outputs).build())
        .layer(3, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation(Activation.SOFTMAX).nIn(num4Inputs).nOut(num4Outputs).build())
        .backprop(true)
        .pretrain(false)
        .build()

      /*创建Mater*/
      val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
        .averagingFrequency(averagingFrequency)
        .storageLevel(StorageLevel.MEMORY_ONLY)
        .workerPrefetchNumBatches(workerPrefetchNumBatches)
        .batchSizePerWorker(batchSizePerWorker)
        .build()

      val sc: SparkContext = SparkUtils.getSparkContext("Spark_DL4J_" + System.currentTimeMillis())
      val sparkNet: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, networkConfiguration, tm)

      val trainList: util.List[DataSet] = new util.ArrayList[DataSet]
      trainList.add(trainDataSet)

      val trainRDD: RDD[DataSet] = sc.parallelize(trainList)
      (1 to iterations).foreach(_ => {
        sparkNet.fit(trainRDD)
      })

      val evaluation: RegressionEvaluation = sparkNet.evaluateRegression(trainRDD)
      println("rmse => " + evaluation.averageMeanSquaredError())
    }
  }
}
