{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE BangPatterns #-}
import AI.Nn.Accelerate
import Data.Array.Accelerate.LLVM.Native (run)
import Data.Array.Accelerate (fromList, Z (..), (:.) (..), use, constant, compute) 
import Data.Monoid
import Control.DeepSeq
import Data.Functor.Identity
import Control.Monad

main :: IO ()
main = do
  {-
  {- Creates a new network with one inputs,
     no hidden layers and one output -}
  network <- new [1, 1]

  {- Train the network for a common logical NOT,
     until the maximum error of 0.01 is reached -}
  let
    trainedNetwork = train run 0.01 network 
      [ (use [0], use [1])
      , (use [1], use [0])
      ]

  {- Predict the learned values -}
    r0 = run (predict trainedNetwork (use [0]))
    r1 = run (predict trainedNetwork (use [1]))

  {- Print the results -}
  putStrLn $ "0 -> " ++ show r0
  putStrLn $ "1 -> " ++ show r1
  -}

  {- Creates a new network with two inputs,
     two hidden layers and one output -}
  network <- new [2, 2, 1]

  putStrLn (replicate 80 '-')
  putStrLn "Random starting network:"
  mapM_ print (fmap neuronWeights network)

  {- Train the network for a common logical AND,
     until the maximum error of 0.01 is reached -}
  let
    trainedNetwork = appEndo (foldMap Endo (replicate 200 (trainUl run samples))) network
      where 
        samples = 
          [ ([0, 0], [0])
          , ([0, 1], [0])
          , ([1, 0], [0])
          , ([1, 1], [1])
          ]
  
  putStrLn (replicate 80 '-')
  putStrLn "Trained network:"
  mapM_ print (fmap neuronWeights trainedNetwork)

  {- Predict the learned values -}
  let
    r00 = run (predict trainedNetwork (use [0, 0]))
    r01 = run (predict trainedNetwork (use [0, 1]))
    r10 = run (predict trainedNetwork (use [1, 0]))
    r11 = run (predict trainedNetwork (use [1, 1]))

  {- Print the results -}
  putStrLn (replicate 80 '-')
  putStrLn "Results:"
  putStrLn $ "0 0 -> " ++ show r00
  putStrLn $ "0 1 -> " ++ show r01
  putStrLn $ "1 0 -> " ++ show r10
  putStrLn $ "1 1 -> " ++ show r11
