@echo off
chcp 65001 >nul
title Пакетный запуск всех работ по Биометрии (Аканов Дархан)

cd /d "%~dp0"

if exist "%~dp0Scripts\activate.bat" (
    call "%~dp0Scripts\activate.bat"
) else (
    if exist "%~dp0.venv\Scripts\activate.bat" (
        call "%~dp0.venv\Scripts\activate.bat"
    )
)

python src\run_labs.py --all
echo.
echo ======================================================================
echo Все работы выполнены! Отчёты и шпаргалка к защите собраны в reports/
echo ======================================================================
pause
