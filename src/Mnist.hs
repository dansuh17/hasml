module Mnist where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import System.Random

img_header_size = 16
label_header_size = 8
img_size = 784

-- coerce n - Integral type
render :: Integral a => a -> Char
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

-- ByteString is an optimized representation of Word8.
-- BS.readFile :: FilePath -> IO ByteString
-- decompress :: ByteString -> ByteString
readMnistAndShow :: IO ()
readMnistAndShow = do
  imgBytes <- decompress <$> BS.readFile "train-images-idx3-ubyte.gz"
  labelBytes <- decompress <$> BS.readFile "train-labels-idx1-ubyte.gz"
  n <- (`mod` 60000) <$> randomIO
  putStr . unlines $
    [(render . BS.index imgBytes . byte_pos n r) <$> [0..27] | r <- [0..27]]
  print $ BS.index labelBytes (n + label_header_size)
    where
      -- find the position of the byte located in (row, col) in the n_th sample
      byte_pos n row col = img_header_size + (n * img_size + row * 28 + col)
