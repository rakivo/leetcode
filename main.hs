module Main where

import Data.Char (isAlpha)

main = undefined

friend :: [String] -> [String]
friend xs = filter (\x -> length x == 4) xs

reverseLetter :: String -> String
reverseLetter xs = reverse $ filter (\c -> isAlpha c) xs
