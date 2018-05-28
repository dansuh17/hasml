{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import NaiveBayes as NB
import StaticTypeNet
-- import UntypedNet
import Mnist (readMnistAndShow, readTrainData)
import Data.Maybe
import System.Environment (getArgs)
import Control.Monad.Random (evalRandIO)
import Text.Read (readMaybe)

main :: IO ()
main = do
  -- readMnistAndShow
  -- now read MNIST and print out the prior distribution
  trainLabeledData <- readTrainData
  putStrLn $ show $ NB.prior trainLabeledData
  putStrLn $ show $ NB.likelihood trainLabeledData

  args <- getArgs
  let n = readMaybe =<< (args !!? 0)
      rate = readMaybe =<< (args !!? 1)
  putStrLn "Training network .."
  putStrLn =<< evalRandIO (netTest (fromMaybe 0.25 rate) (fromMaybe 500000 n))
