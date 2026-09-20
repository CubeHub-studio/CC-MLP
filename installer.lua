-- CC-MLP installer
-- Downloads the complete CC-MLP runtime from GitHub.

local base="https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/"
local files={
 "mlp.lua","example.lua","math_example.lua",
 "tokenizer.lua","tinygpt.lua","train_gpt.lua","chat_gpt.lua",
 "GPT_README.md","README.md","LICENSE"
}

print("CC-MLP Installer")
print("Downloading "..#files.." files...")

for i,name in ipairs(files) do
 print("["..i.."/"..#files.."] "..name)
 local ok=shell.run("wget",base..name,name)
 if not ok then
  printError("Failed to download "..name)
  return
 end
end

print("CC-MLP installed successfully!")
print("Run 'example.lua' for the MLP XOR demo.")
print("Run 'chat_gpt.lua' after training CC-GPT.")
