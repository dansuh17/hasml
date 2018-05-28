{-# LANGUAGE ScopedTypeVariables #-}

module NaiveBayes where

import Mnist (LabeledData, groupByLabel, dat)
import Numeric.LinearAlgebra (Vector, Matrix, R, size, fromList, toColumns, sumElements, fromRows)

-- calculates the prior probability of a dataset
-- the data are organized as rows having same label
-- P(y)
prior :: LabeledData -> Vector R
prior ld = fromList $ map ((/ fromIntegral totalNum) . fromIntegral) numElemsPerLabel
  where
    grouped :: [Matrix R] = groupByLabel ld
    numElemsPerLabel :: [Int] = map (fst . size) grouped
    totalNum = sum numElemsPerLabel

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
    attrProb classSum = fromList $ map (\ a -> fst a / snd a) $ zip classSum totalSum
