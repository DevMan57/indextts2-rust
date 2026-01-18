@echo off
REM IndexTTS2-Rust Setup Script
REM Run this in Git Bash or WSL, not PowerShell

echo ========================================
echo IndexTTS2-Rust GitHub Setup
echo ========================================

cd /d "C:\AI\indextts2-rust"

echo.
echo [1/5] Initializing git repository...
git init

echo.
echo [2/5] Adding all files...
git add .

echo.
echo [3/5] Creating initial commit...
git commit -m "Initial commit: IndexTTS2 Rust rewrite project structure"

echo.
echo [4/5] Setting up remote...
git branch -M main
git remote add origin https://github.com/DevMan57/indextts2-rust.git

echo.
echo [5/5] Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo Setup complete!
echo.
echo Repository: https://github.com/DevMan57/indextts2-rust
echo ========================================

pause
