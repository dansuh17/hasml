module Mnist where

import Data.List (nub)
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Numeric.LinearAlgebra (vector, Vector, R, Z, reshape, Matrix, toList, toRows, fromRows)

-- | typeclass that represents a dataset
class DataSet a where
  groupByLabel :: a -> [Matrix R]
  distinctLabels :: a -> [R]
  -- oneHotLabel :: a -> Matrix Z

img_header_size = 16
label_header_size = 8
img_size = 784

-- labeled data consists of data and their label
data LabeledData = LabeledData
    { dat :: !(Matrix R)  -- this has num_data >< data_dimension
    , label :: !(Vector R)
    }

-- define LabeledData as instance of DataSet typeclass
instance DataSet LabeledData where
    distinctLabels = nub . toList . label
    groupByLabel ld = [
        -- extract index that matches the label, and cherrypicks them from data rows
        fromRows [dataRows !! idx | idx <- extractIdx lab]
        | lab <- distinctLabels ld ]
      where
        dataRows = toRows $ dat ld  -- represent matrix as list of rows
        labels = toList $ label ld  -- represent vector to list of values
        labelIdx = zip labels [0..]  -- zip it with index
        extractIdx targetLabel = map snd $ filter ((== targetLabel) . fst) labelIdx

    -- groupByLabel ld = [ | lab <- ]

-- mnist dataset consists of train set and a test set
data Mnist = Mnist
    { trainData :: LabeledData
    , testData :: LabeledData
    }

-- coerce n - Integral type
render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

-- convert byteString to vector -- in this case, bytestring is read as numbers byte by byte
byteStringToVector :: BS.ByteString -> Vector R
byteStringToVector = vector . map fromIntegral . BS.unpack

readDataFile :: String -> IO BS.ByteString
readDataFile filepath = decompress <$> BS.readFile filepath

-- make label data represented as Vector
makeLabelData :: BS.ByteString -> Vector R
makeLabelData = byteStringToVector . BS.drop 8  -- 8-byte header

-- make image data represented as Matrix
makeImgData :: BS.ByteString -> Matrix R
makeImgData = reshape img_size . byteStringToVector . BS.drop 16  -- 16-byte header

readTrainData :: IO LabeledData
readTrainData = do
  imgData <- makeImgData <$> decompress <$> BS.readFile "train-images-idx3-ubyte.gz"
  labelData <- makeLabelData <$> decompress <$> BS.readFile "train-labels-idx1-ubyte.gz"
  return $ LabeledData imgData labelData

-- ByteString is an optimized representation of Word8.
-- BS.readFile :: FilePath -> IO ByteString
-- decompress :: ByteString -> ByteString
readMnistAndShow :: IO ()
readMnistAndShow = do
  imgData <- makeImgData <$> decompress <$> BS.readFile "train-images-idx3-ubyte.gz"
  labelData <- makeLabelData <$> decompress <$> BS.readFile "train-labels-idx1-ubyte.gz"
  putStrLn $ show $ labelData
  -- putStr $ show $ makeImgData imgBytes
{-
  n <- (`mod` 60000) <$> randomIO
  putStr . unlines $
    [(render . BS.index imgBytes . byte_pos n r) <$> [0..27] | r <- [0..27]]
  print $ BS.index labelBytes (n + label_header_size)
    where
      -- find the position of the byte located in (row, col) in the n_th sample
      byte_pos n row col = img_header_size + (n * img_size + row * 28 + col)
      -}

