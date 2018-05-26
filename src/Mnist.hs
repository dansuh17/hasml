module Mnist where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import System.Random
import Numeric.LinearAlgebra (vector, Vector, R, Z, reshape, Matrix)

img_header_size = 16
label_header_size = 8
img_size = 784

data MnistData = Mnist { trainImg :: Matrix R
                       , trainLabel :: Vector R
                       , testImage :: Matrix R
                       , testLabel :: Vector R
                       }

-- coerce n - Integral type
render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

-- convert byteString to vector -- in this case, bytestring is read as numbers byte by byte
byteStringToVector :: BS.ByteString -> Vector R
byteStringToVector = vector . map fromIntegral . BS.unpack

readDataFile :: String -> IO BS.ByteString
readDataFile filepath = decompress <$> BS.readFile filepath

makeLabelData :: BS.ByteString -> Vector R
makeLabelData = byteStringToVector . BS.drop 8

makeImgData :: BS.ByteString -> Matrix R
makeImgData = reshape img_size . byteStringToVector . BS.drop 16

-- ByteString is an optimized representation of Word8.
-- BS.readFile :: FilePath -> IO ByteString
-- decompress :: ByteString -> ByteString
readMnistAndShow :: IO ()
readMnistAndShow = do
  imgBytes <- decompress <$> BS.readFile "train-images-idx3-ubyte.gz"
  labelBytes <- decompress <$> BS.readFile "train-labels-idx1-ubyte.gz"
  putStr $ show $ makeLabelData labelBytes
  -- putStr $ show $ reshape img_size $ byteStringToVector (BS.drop 16 imgBytes)
{-
  n <- (`mod` 60000) <$> randomIO
  putStr . unlines $
    [(render . BS.index imgBytes . byte_pos n r) <$> [0..27] | r <- [0..27]]
  print $ BS.index labelBytes (n + label_header_size)
    where
      -- find the position of the byte located in (row, col) in the n_th sample
      byte_pos n row col = img_header_size + (n * img_size + row * 28 + col)
      -}

