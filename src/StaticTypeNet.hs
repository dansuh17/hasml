{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeOperators #-}

module StaticTypeNet where

import GHC.TypeLits
import Control.Monad
import Control.Monad.Random
import Data.List
import Data.Maybe
import Numeric.LinearAlgebra.Static (L, R, randomVector, RandDist(Uniform), uniformSample)
import System.Environment
import Text.Read

-- Weights :: Nat -> Nat -> *
data Weights i o = W { wBiases :: !(R o)  -- dependent type : i Doubles
                     , wNodes :: !(L o i)  -- o x i vector of Doubles
                     }   -- and o x i layer

-- Nat = type-level numeral
-- takes two types of kind Nat and returns a * (normal type)
-- [Nat] is a type-level list of Nats
-- optional ' (apostrophe) is for distinguishing from normal lists
data Network :: Nat -> [Nat] -> Nat -> * where
  O :: !(Weights i o)
    -> Network i '[] o
  (:&~) :: KnownNat h
        => !(Weights i h)
        -> !(Network h hs o)
        -> Network i (h':hs) o

infixr 5 :&~

-- random weight
randomWeights :: (MonadRandom m, KnownNat i, KnownNat o) => m (Weights i o)
randomWeights = do
  s1 :: Int <- getRandom
  s2 :: Int <- getRandom
  let wB = randomVector s1 Uniform * 2 - 1  -- uses type inference to determine the input / output sizes!
      wN = uniformSample s2 (-1) 1
  return $ W wB wN
