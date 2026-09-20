-- CC-MLP updater
-- Re-downloads every installed CC-MLP file from GitHub.

local base="https://raw.githubusercontent.com/CubeHub-studio/CC-MLP/main/"
local files={
 "mlp.lua","example.lua","math_example.lua",
 "tokenizer.lua","tinygpt.lua","train_gpt.lua","chat_gpt.lua",
 "GPT_README.md","README.md","LICENSE",
 "installer.lua","updater.lua"
}

print("CC-MLP Updater")
print("Checking "..#files.." files...")

for i,name in ipairs(files) do
 print("["..i.."/"..#files.."] Updating "..name)
 local ok=shell.run("wget",base..name,name)
 if not ok then
  printError("Failed to update "..name)
  return
 end
end

print("CC-MLP updated successfully!")
