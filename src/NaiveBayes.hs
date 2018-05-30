{-# LANGUAGE ScopedTypeVariables #-}

module NaiveBayes where

import Data.List (maximumBy)
import Data.Ord (comparing)
import Mnist (LabeledData, groupByLabel, dat)
import Data.Function (on)
import Numeric.LinearAlgebra (Vector, Matrix, R, size, fromList, toColumns, sumElements, fromRows, (#>), toList)

-- TODO: later expand this by receiving label type and data type
data NaiveBayesClassifier = NBC
    { p :: !(Vector R)
    , l :: !(Matrix R)
    }

-- represents a classifier that predicts the class from input sample
class Classifier a where
    -- given the prior and likelihood, with a sample data, predict the class
    predict :: a -> Vector R -> Int  -- TODO: parameterize label type

instance Classifier NaiveBayesClassifier where
    predict nbc sample = argmax $ (p nbc) * (l nbc #> sample)

-- divide two integers to make a fraction
fractionDiv :: Fractional b => Int -> Int -> b
fractionDiv = (/) `on` fromIntegral

-- calculates the prior probability of a dataset
-- the data are organized as rows having same label
-- P(y)
prior :: LabeledData -> Vector R
prior ld = fromList $ map (flip fractionDiv totalNum) numElemsPerLabel
  where
    grouped :: [Matrix R] = groupByLabel ld
    numElemsPerLabel :: [Int] = map (fst . size) grouped
    totalNum = sum numElemsPerLabel

-- used as prior distribution of likelihood
epsilon :: R = 0.001

-- conditional likelihood of each data attributes given the class
-- P(x | y)
likelihood :: LabeledData -> Matrix R  -- data_class * data_dimension shape
likelihood ld = fromRows $ map attrProb colSumByLabel
  where
    grouped :: [Matrix R] = groupByLabel ld
    colSum :: Matrix R -> [R] = map sumElements . toColumns
    colSumByLabel :: [[R]] = map colSum grouped
    totalSum :: [R] = colSum $ dat ld
    -- create a vector of size DATA_DIMENSION that contains probability of each attributes
    attrProb classSum = fromList $ map (\ a -> fst a / (epsilon + snd a)) $ zip classSum totalSum

-- find maximum index in the vector
argmax :: Vector R -> Int
argmax = fst . maximumBy (comparing snd) . zip [0..] . toList
