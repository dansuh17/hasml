{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Main where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import System.Random
import Numeric.LinearAlgebra (Matrix, Vector, randomVector, uniformSample, RandDist(Uniform), (#>))
import Control.Monad.Random (MonadRandom, getRandom)

img_header_size = 16
label_header_size = 8
img_size = 784

-- coerce n - Integral type
render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

-- ByteString is an optimized representation of Word8.
-- BS.readFile :: FilePath -> IO ByteString
-- decompress :: ByteString -> ByteString
main :: IO ()
main = do
  imgBytes <- decompress <$> BS.readFile "train-images-idx3-ubyte.gz"
  labelBytes <- decompress <$> BS.readFile "train-labels-idx1-ubyte.gz"
  n <- (`mod` 60000) <$> randomIO
  putStr . unlines $
    [(render . BS.index imgBytes . byte_pos n r) <$> [0..27] | r <- [0..27]]
  print $ BS.index labelBytes (n + label_header_size)
    where
      -- find the position of the byte located in (row, col) in the n_th sample
      byte_pos n row col = img_header_size + (n * img_size + row * 28 + col)

-- Wx + b
data Weights = W { wBiases :: !(Vector Double)  -- n
                 , wNodes :: !(Matrix Double)  -- n x m
                 }  -- "m to n" layer

-- define a network data type
-- can build a network like : ih :&~ hh :&~ O ho
data Network :: * where
    O :: !Weights -> Network
    (:&~) :: !Weights -> !Network -> Network

infixr 5 :&~

{- equivalent to:
data Network = O Weights
             | Weights :&~ Network
-}

randomWeights :: MonadRandom m => Int -> Int -> m Weights
randomWeights i o = do
  seed1 :: Int <- getRandom  -- generate random seed
  seed2 :: Int <- getRandom
      -- randomVector :: Seed -> RandDist -> Int -> VectorDouble
      -- obtains a vector of random elements of provided size
  let wB = randomVector seed1 Uniform o * 2 - 1
      -- uniformSample :: Seed -> Int -> [(Double, Double)] -> Matrix Double
      wN = uniformSample seed2 o (replicate i (-1, 1))  -- o x i -sized matrix
  return $ W wB wN

-- build a random network composed of layers
-- input layer size -> [hidden layer sizes] -> output layer size
randomNet :: MonadRandom m => Int -> [Int] -> Int -> m Network
randomNet i [] o = O <$> randomWeights i o  -- final fully-connected layer
randomNet i (h:hs) o = (:&~) <$> randomWeights i h <*> randomNet h hs o

-- the logistic (sigmoid) function
logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

-- matrix-vector multiplication
-- (#>) :: Numeric t => Matrix t -> Vector t -> Vector t
runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + wN #> v

-- feedforward the network with every activaciton function as sigmoid
runNet :: Network -> Vector Double -> Vector Double
runNet (O w) !v = logistic (runLayer w v)
runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
                       in runNet n' v'
