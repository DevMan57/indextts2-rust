#!/bin/bash
# Run this in Git Bash

cd /c/AI/indextts2-rust

echo "========================================"
echo "IndexTTS2-Rust GitHub Push"
echo "========================================"

# Configure git identity
git config user.email "devman57@users.noreply.github.com"
git config user.name "DevMan57"

# Add all files
echo "[*] Adding files..."
git add .

# Commit
echo "[*] Creating commit..."
git commit -m "Initial commit: IndexTTS2 Rust rewrite with Ralph + MCP setup"

# Setup remote
git remote remove origin 2>/dev/null
git remote add origin https://github.com/DevMan57/indextts2-rust.git

# Push
echo "[*] Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "========================================"
echo "Done! https://github.com/DevMan57/indextts2-rust"
echo "========================================"
