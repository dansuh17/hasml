{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module UntypedNet where

import Data.Functor
import Data.Maybe
import Data.List (foldl')
import Numeric.LinearAlgebra (Matrix, Vector, randomVector, uniformSample, RandDist(Uniform), (#>), tr, outer, scale, norm_2, vector)
import Control.Monad.Random (MonadRandom, getRandom, replicateM)

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

-- derivative function of logistic
logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

-- matrix-vector multiplication
-- (#>) :: Numeric t => Matrix t -> Vector t -> Vector t
runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + wN #> v

-- feedforward the network with every activaciton function as sigmoid
runNet :: Network -> Vector Double -> Vector Double
runNet (O w) !v = logistic (runLayer w v)
runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
                       in runNet n' v'

train :: Double  -- learning rate
      -> Vector Double  -- input vector
      -> Vector Double  -- target vector
      -> Network  -- network to train
      -> Network
train rate x0 target = fst . go x0
  where
    go :: Vector Double  -- input vector
       -> Network  -- network to train
       -> (Network, Vector Double)  -- (new network layer, back-propping derivative chain)
    -- handle the output layer
    go !x (O w@(W wB wN))
      = let y = runLayer w x
            o = logistic y
            -- the gradient (dE / dy)
            dEdy = logistic' y * (o - target)
            -- bias and node update
            wB' = wB - scale rate dEdy  -- multiply by scalar 'rate'
            wN' = wN - scale rate (dEdy `outer` x)
            w' = W wB' wN'
            -- bundle of derivatives for next step
            dWs = tr wN #> dEdy
         in (O w', dWs)
    -- handle the inner layers
    go !x (w@(W wB wN) :&~ n)
        = let y = runLayer w x
              o = logistic y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy = logistic' y * dWs'
              -- new bias weights and node weights
              wB' = wB - scale rate dEdy
              wN' = wN - scale rate (dEdy `outer` x)
              w' = W wB' wN'
              -- bundle of derivatives for next step
              dWs = tr wN #> dEdy
          in (w' :&~ n', dWs)

netTest :: MonadRandom m => Double -> Int -> m String
netTest rate n = do
    inps <- replicateM n $ do
      s <- getRandom
      return $ randomVector s Uniform 2 * 2 - 1
    let outs = flip map inps $ \v ->
                 if v `inCircle` (fromRational 0.33, 0.33)
                      || v `inCircle` (fromRational (-0.33), 0.33)
                   then fromRational 1
                   else fromRational 0
    net0 <- randomNet 2 [16,8] 1
    let trained = foldl' trainEach net0 (zip inps outs)
          where
            trainEach :: Network -> (Vector Double, Vector Double) -> Network
            trainEach nt (i, o) = train rate i o nt

        outMat = [ [ render (norm_2 (runNet trained (vector [x / 25 - 1,y / 10 - 1])))
                   | x <- [0..50] ]
                 | y <- [0..20] ]
        render r | r <= 0.2  = ' '
                 | r <= 0.4  = '.'
                 | r <= 0.6  = '-'
                 | r <= 0.8  = '='
                 | otherwise = '#'

    return $ unlines outMat
  where
    inCircle :: Vector Double -> (Vector Double, Double) -> Bool
    v `inCircle` (o, r) = norm_2 (v - o) <= r

(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)
