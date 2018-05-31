{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import NaiveBayes as NB
import StaticTypeNet
import Mnist (readMnistAndShow, readData, dat, label, readMnist, trainData, testData, LabeledData, dataLabel, render)
import Data.Maybe
import System.Environment (getArgs)
import Control.Monad.Random (evalRandIO)
import Text.Read (readMaybe)
import Numeric.LinearAlgebra (Vector, R, toRows, fromList, toList)

main :: IO ()
main = do
  -- readMnistAndShow
  -- now read MNIST and print out the prior distribution
  mnist <- readMnist
  let trainLabeledData :: LabeledData = trainData mnist  -- train set
      testLabeledData :: LabeledData = testData mnist  -- test set
      pri = NB.prior trainLabeledData
      lk = NB.likelihood trainLabeledData
      naiveBayesClassifier :: NB.NaiveBayesClassifier = NB.NBC pri lk  -- create a classifier
      predictor = predict naiveBayesClassifier

  -- show the prior
  putStrLn $ "Below is the prior distribution"
  putStrLn $ show pri  -- show prior values

  let datlab :: [(Vector R, R)] = dataLabel testLabeledData  -- data-label pairs
      predictedAndTruth ld = (predictor (fst ld), floor $ snd ld)
      -- the result of prediction
      predictedResult :: [(Int, Int)] = map predictedAndTruth datlab
      -- calculate the accuracy
      accuracy :: [(Int, Int)] -> Double
      accuracy res = NB.fractionDiv
          (foldl
              (\acc elem -> if fst elem == snd elem
                            then acc + 1
                            else acc)
              0 res)  -- accumulate correct predictions
          (length res)  -- ...divided by total size

  -- show predicted value and truth label
  mapM_ (putStrLn . show . predictedAndTruth) datlab

  putStrLn $ "The accuracy is: "
  putStrLn $ show $ accuracy predictedResult

  -- show the 100th sample
  putStrLn $ render $ fst $ datlab !! 100
  putStrLn $ show $ predictedResult !! 100

  args <- getArgs
  let n = readMaybe =<< (args !!? 0)
      rate = readMaybe =<< (args !!? 1)
  putStrLn "Training network .."
  putStrLn =<< evalRandIO (netTest (fromMaybe 0.25 rate) (fromMaybe 500000 n))
