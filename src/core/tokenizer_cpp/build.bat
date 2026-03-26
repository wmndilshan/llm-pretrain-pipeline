@echo off
REM Build script for C++ tokenizer on Windows
REM Requires: Visual Studio Build Tools or MinGW

echo Building C++ Tokenizer...

REM Try MinGW first (g++)
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MinGW g++...
    g++ -O3 -shared -fPIC -fopenmp -o tokenizer.dll tokenizer.cpp -static-libgcc -static-libstdc++
    if %ERRORLEVEL% EQU 0 (
        echo Build successful: tokenizer.dll
        exit /b 0
    )
)

REM Try MSVC
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MSVC cl...
    cl /O2 /LD /openmp /EHsc tokenizer.cpp /Fe:tokenizer.dll
    if %ERRORLEVEL% EQU 0 (
        echo Build successful: tokenizer.dll
        exit /b 0
    )
)

echo ERROR: No C++ compiler found!
echo Please install MinGW-w64 or Visual Studio Build Tools
exit /b 1
