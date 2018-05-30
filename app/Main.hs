{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import NaiveBayes as NB
import StaticTypeNet
-- import UntypedNet
import Mnist (readMnistAndShow, readData, dat, label, readMnist, trainData, testData)
import Data.Maybe
import System.Environment (getArgs)
import Control.Monad.Random (evalRandIO)
import Text.Read (readMaybe)
import Numeric.LinearAlgebra (toRows, fromList, toList)

main :: IO ()
main = do
  -- readMnistAndShow
  -- now read MNIST and print out the prior distribution
  mnist <- readMnist
  let trainLabeledData = trainData mnist
      testLabeledData = testData mnist
      pri = NB.prior trainLabeledData
      lk = NB.likelihood trainLabeledData
      predictor = predict pri lk

  -- show the prior
  putStrLn $ "Below is the prior distribution"
  putStrLn $ show pri  -- show prior values

  let testData = toRows $ dat testLabeledData
      testLabel = toList $ label testLabeledData
      dataLabel = zip testData testLabel
      predictedAndTruth ld = (predictor (fst ld), floor $ snd ld)
      predictedResult :: [(Int, Int)] = map predictedAndTruth dataLabel
      accuracy :: [(Int, Int)] -> Double
      accuracy res = NB.fractionDiv (foldl (\acc elem -> if fst elem == snd elem then acc + 1 else acc) 0 res) (length res)

  -- show predicted value and truth label
  mapM_ (putStrLn . show . predictedAndTruth) dataLabel
  putStrLn $ "The accuracy is: "
  putStrLn $ show $ accuracy predictedResult

  args <- getArgs
  let n = readMaybe =<< (args !!? 0)
      rate = readMaybe =<< (args !!? 1)
  putStrLn "Training network .."
  putStrLn =<< evalRandIO (netTest (fromMaybe 0.25 rate) (fromMaybe 500000 n))
