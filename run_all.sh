#!/usr/bin/env bash
# Универсальный запуск всех лабораторных работ по биометрии на Linux / macOS
# Студент: Аканов Дархан | Группа: SIS-2309 | ID: 38041

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Активация виртуального окружения, если оно существует
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/bin/activate" ]; then
    source "$SCRIPT_DIR/bin/activate"
fi

echo "======================================================================"
echo "Лабораторные работы по Биометрии (Аканов Дархан, SIS-2309, ID 38041)"
echo "Репозиторий: https://github.com/Dark-Ak06/Biometric"
echo "======================================================================"

python3 src/run_labs.py "$@"
