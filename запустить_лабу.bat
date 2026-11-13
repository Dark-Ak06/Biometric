@echo off
chcp 65001 >nul
title Лабораторные работы по Биометрии (Аканов Дархан SIS-2309)

cd /d "%~dp0"

if exist "%~dp0Scripts\activate.bat" (
    call "%~dp0Scripts\activate.bat"
) else (
    if exist "%~dp0.venv\Scripts\activate.bat" (
        call "%~dp0.venv\Scripts\activate.bat"
    )
)

python src\run_labs.py %*
if errorlevel 1 pause
