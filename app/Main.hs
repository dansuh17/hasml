{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import UntypedNet
import Mnist (readMnistAndShow)
import Data.Maybe
import System.Environment (getArgs)
import Control.Monad.Random (evalRandIO)
import Text.Read (readMaybe)

main :: IO ()
main = do
  readMnistAndShow
  args <- getArgs
  let n = readMaybe =<< (args !!? 0)
      rate = readMaybe =<< (args !!? 1)
  putStrLn "Training network .."
  putStrLn =<< evalRandIO (netTest (fromMaybe 0.25 rate) (fromMaybe 500000 n))
