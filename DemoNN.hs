import AI.Nn
import Data.Monoid

main :: IO ()
main = do
  {- Creates a new network with two inputs,
     two hidden layers and one output -}
  network <- new [2, 2, 1]

  print network

  {- Train the network for a common logical AND,
     until the maximum error of 0.01 is reached -}
  let trainedNetwork = train 0.001 network [([0, 0], [0])
                                           ,([0, 1], [0])
                                           ,([1, 0], [0])
                                           ,([1, 1], [1])]

  print trainedNetwork

  {- Predict the learned values -}
  let r00 = predict trainedNetwork [0, 0]
      r01 = predict trainedNetwork [0, 1]
      r10 = predict trainedNetwork [1, 0]
      r11 = predict trainedNetwork [1, 1]

  {- Print the results -}
  putStrLn $ "0 0 -> " ++ show r00
  putStrLn $ "0 1 -> " ++ show r01
  putStrLn $ "1 0 -> " ++ show r10
  putStrLn $ "1 1 -> " ++ show r11
