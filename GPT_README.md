# CC-GPT

A tiny educational decoder-only Transformer written in Lua for CC:Tweaked.

Architecture:
- 128-token byte/ASCII vocabulary
- token embeddings
- causal self-attention
- residual connections
- ReLU feed-forward network
- vocabulary projection and softmax
- autoregressive next-token generation
- gradient-based training

Default size:
- embedding size: 16
- feed-forward size: 32
- one attention block
- 32-token context

This is a real Transformer-style language model, but it is vastly smaller than ChatGPT and is intended for learning and experimentation.

Install these files with wget:
wget https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/tokenizer.lua
wget https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/tinygpt.lua
wget https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/train_gpt.lua
wget https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/chat_gpt.lua

Create training.txt with some text, then run:
train_gpt.lua training.txt 10 32

Then:
chat_gpt.lua

API:
local GPT=require("tinygpt")
local tok=require("tokenizer")
local model=GPT.new({d_model=16,ff_dim=32,max_seq=32,learning_rate=0.01})
local tokens=tok.encode("hello")
local probabilities=model:predict(tokens)
local generated=model:generate(tokens,64,0.8)
model:save("tiny-gpt.model")
local loaded=GPT.load("tiny-gpt.model")

Files:
tokenizer.lua
tinygpt.lua
train_gpt.lua
chat_gpt.lua
GPT_README.md
