{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import NaiveBayes as NB
import StaticTypeNet
-- import UntypedNet
import Mnist (readMnistAndShow, readTrainData, dat)
import Data.Maybe
import System.Environment (getArgs)
import Control.Monad.Random (evalRandIO)
import Text.Read (readMaybe)
import Numeric.LinearAlgebra (toRows, fromList)

main :: IO ()
main = do
  -- readMnistAndShow
  -- now read MNIST and print out the prior distribution
  trainLabeledData <- readTrainData
  let pri = NB.prior trainLabeledData
      lk = NB.likelihood trainLabeledData
      predictor = predict pri lk
  putStrLn $ show pri  -- show prior values
  putStrLn $ show $ predictor $ (toRows $ dat trainLabeledData) !! 2

  args <- getArgs
  let n = readMaybe =<< (args !!? 0)
      rate = readMaybe =<< (args !!? 1)
  putStrLn "Training network .."
  putStrLn =<< evalRandIO (netTest (fromMaybe 0.25 rate) (fromMaybe 500000 n))
