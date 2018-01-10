package com.jerry.vehicle.utils

import java.io.File
import java.lang.reflect.Type
import java.util

import com.alibaba.fastjson.{JSON, TypeReference}

import scala.collection.JavaConversions._
import scala.io.{BufferedSource, Source}

object JsonUtils {
  /**
    * 从json文件中读出特征映射
    *
    * @param path
    * @return
    */
  def getTypeMapping(path: String): util.Map[Int, util.Map[String, Int]] = {
    val reader: BufferedSource = Source.fromFile(new File(path))
    val json: String = reader.bufferedReader().readLine()
    val typeReference: Type = new TypeReference[util.Map[Int, util.Map[String, Int]]]() {}.getType
    val map: util.Map[Int, util.Map[String, Int]] = JSON.parseObject(json, typeReference)
    map
  }

  /**
    * 获得反向映射
    *
    * @param mapping
    * @return
    */
  def getTypeMapping(mapping: util.Map[Int, util.Map[String, Int]]): util.Map[Int, util.Map[Int, String]] = {
    val map = new util.HashMap[Int, util.Map[Int, String]]()
    for ((index: Int, son: util.Map[String, Int]) <- mapping) {
      val daughter: util.Map[Int, String] = new util.HashMap[Int, String]()
      for ((typeString: String, typeCode: Int) <- son) {
        daughter.put(typeCode, typeString)
      }
      map.put(index, daughter)
    }
    map
  }
}
