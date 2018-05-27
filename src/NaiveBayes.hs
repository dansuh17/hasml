module NaiveBayes where

import Numeric.LinearAlgebra (Vector, Matrix, R)

-- calculates the prior probability of a dataset
-- the data are organized as rows having same label
-- prior :: LabeledData -> Matrix R
