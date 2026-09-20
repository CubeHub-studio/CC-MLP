# CC-MLP

A small dependency-free Multilayer Perceptron library for CC:Tweaked.

Features:
- Arbitrary layer sizes
- Backpropagation
- Sigmoid, tanh, and ReLU hidden activations
- Configurable output activation
- Batch/epoch training
- Prediction and binary accuracy
- Save/load trained networks
- Pure Lua

Install:
Put mlp.lua in the computer filesystem and use require("mlp").

Example:
local MLP=require("mlp")
local brain=MLP.new({2,4,4,1},{learning_rate=0.5,activation="tanh",output_activation="sigmoid"})
local output=brain:predict({0,1})

Training samples are {{input...},{target...}}.

The included example.lua trains XOR using a 2-4-4-1 network.
math_example.lua demonstrates regression.

API:
MLP.new(sizes, options)
brain:predict(input)
brain:train(input,target)
brain:train_batch(samples,epochs)
brain:accuracy(samples,threshold)
brain:save(path)
MLP.load(path)

MIT License.
