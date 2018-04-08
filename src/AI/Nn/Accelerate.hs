{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE LambdaCase #-}
-- | This module contains everything related to the main library interface
--
-- @since 0.1.0

module AI.Nn.Accelerate where

import qualified Prelude as P
import qualified Data.Foldable as P
import Data.Array.Accelerate.Numeric.LinearAlgebra
import Data.Array.Accelerate
import Data.Array.Accelerate.System.Random.MWC
import AI.Nn (Activation (..))

-- | The network
type Network = Network' ()

-- | The alias for a list of layers
type Network' a = [Layer a]

-- | A network layer
data Layer a = Layer
  { neuronWeights :: Matrix Double -- ^ The matrix of the neuron weights
  , activation    :: Activation
  , extraInfo     :: a -- ^ Usually either () or Forward
  }

-- | The forward layer type
data Forward = Forward
  { outputs         :: Vector Double -- ^ The outputs of the neurons in this layer
  , sumInputWeights :: Vector Double -- ^ The sum of the input weights for all neurons in this layer
  , inputs          :: Vector Double -- ^ The inputs to this layer
  }

activate, activate' :: Activation -> Exp Double -> Exp Double
activate a = case a of
  Sigmoid -> sigmoid
  Output -> P.id
activate' a = case a of
  Sigmoid -> sigmoid'
  Output -> const 1

-- | The sigmoid activation function
sigmoid :: Exp Double -> Exp Double
sigmoid x = 1 / (1 + exp (-x))

-- | The first derivation of the sigmoid function
sigmoid' :: Exp Double -> Exp Double
sigmoid' x = let y = sigmoid x in y * (1 - y)

reLU :: Exp Double -> Exp Double
reLU = max 0

reLU' :: Exp Double -> Exp Double
reLU' x = cond (x < 0) 0 1

-- | Create a new sigmoid Layer from a list of Neuron'
sigmoidLayer :: Matrix Double -> Layer ()
sigmoidLayer weights = Layer weights Sigmoid ()

-- | Create a new output Layer from a list of Neuron'
outputLayer :: Matrix Double -> Layer ()
outputLayer n = Layer n Output ()

new :: [Int] -> P.IO Network
new xs = do
  xs' <- P.mapM newLayer (P.zip xs (P.tail xs))
  P.pure (P.map sigmoidLayer (P.init xs') P.++ [outputLayer (P.last xs')])

newLayer :: (Int,Int) -> P.IO (Matrix Double)
newLayer (i, o) = randomArray (uniformR (-1,1)) (Z :. o :. i + 1)
  -- one extra column for the bias

type RunFn = forall a . Arrays a => Acc a -> a

-- | Do the complete back propagation
backpropagate :: RunFn -> Network -> (Vector Double, Vector Double) -> Network
backpropagate run nw (xs, ys) = weightUpdate run (forwardLayer run nw (use xs)) (use ys)

-- | The learning rate
rate :: Exp Double
rate = 0.5

addBias :: Acc (Vector Double) -> Acc (Vector Double)
addBias = (fill (constant (Z:.1)) 1 ++)

-- | Generate forward pass info
forwardLayer :: RunFn -> Network -> Acc (Vector Double) -> Network' Forward
forwardLayer run nw xs = P.reverse (P.fst (P.foldl' pf ([], xs) nw))
  where
    pf :: (Network' Forward, Acc (Vector Double)) -> Layer () -> (Network' Forward, Acc (Vector Double))
    pf (nw', xs') a = let y = propagate xs' a in (y : nw', use (outputs (extraInfo y)))

    propagate :: Acc (Vector Double) -> Layer () -> Layer Forward
    propagate prev next = next { extraInfo = mkForward prev next }

    mkForward :: Acc (Vector Double) -> Layer () -> Forward
    mkForward inps layer = Forward (run (map (activate (activation layer)) (use out))) out (run inps)
      where
        out = run (use (neuronWeights layer) #> addBias inps)

-- | Updates the weights for an entire network
weightUpdate
  :: RunFn
  -> Network' Forward
  -> Acc (Vector Double) -- ^ desired output value
  -> Network
weightUpdate run fpnw ys = P.fst $ P.foldr (updateLayer run) ([], ds) fpnw
  where ds = zipWith (-) ys (use (outputs (extraInfo (P.last fpnw))))

-- | Updates the weights for a layer
updateLayer :: RunFn -> Layer Forward -> (Network, Acc (Vector Double)) -> (Network, Acc (Vector Double))
updateLayer run fpl (nw, ds) = (l : nw, ds')
 where
  Z:.h:.w = unlift (shape (use (neuronWeights fpl))) :: Z:.Exp Int:.Exp Int

  l :: Layer ()
  l = fpl { neuronWeights = run (zipWith3 (\weight x e -> weight + rate * e * x)
              (use (neuronWeights fpl))
              (replicate (lift (Z:.h:.All)) (addBias (use (inputs (extraInfo fpl)))))
              (replicate (lift (Z:.All:.w)) es))
          , extraInfo = ()
          }

  es :: Acc (Vector Double)
  es = zipWith (\out d -> activate' (activation fpl) out * d) (use (sumInputWeights (extraInfo fpl))) ds

  ds' = tail $ es <# use (neuronWeights l)

-- | Create an indefinite sequence of networks
trainUl :: RunFn -> [(Vector Double, Vector Double)] -> Network -> Network
trainUl run samples nw = P.foldl' (backpropagate run) nw samples

-- | Quadratic error for multiple pairs
globalQuadError :: Network -> [(Acc (Vector Double), Acc (Vector Double))] -> Exp Double
globalQuadError nw samples = P.foldr (+) 0 (P.map (quadErrorNet nw) samples)

-- | Quadratic error for a single vector pair
quadErrorNet :: Network -> (Acc (Vector Double), Acc (Vector Double)) -> Exp Double
quadErrorNet nw (xs, ys) =
  the (sum (zipWith (\o y -> (y - o) ** 2) (predict nw xs) ys))

-- | Calculates the output of a network for a given input vector
predict :: Network -> Acc (Vector Double) -> Acc (Vector Double)
predict nw xs = P.foldl calculateLayer xs nw
  where
   calculateLayer :: Acc (Vector Double) -> Layer () -> Acc (Vector Double)
   calculateLayer s n = map (activate (activation n)) (use (neuronWeights n) #> addBias s)
