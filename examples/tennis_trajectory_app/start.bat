@echo off
title Tennis Trajectory Detection Engine
echo ===============================================
echo Tennis Trajectory Detection Engine - YOLOv8n
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found! Current version:
python --version

REM Check if we're in the right directory
if not exist "core\__init__.py" (
    echo ERROR: Please run this script from the tennis_trajectory_app directory
    echo Expected structure: tennis_trajectory_app\core\__init__.py
    pause
    exit /b 1
)

echo.
echo Starting Tennis Trajectory Detection Engine...
echo.

REM Show options
echo Available options:
echo 1. Run GUI interface (with model setup)
echo 2. Run CLI interface
echo 3. Run demo
echo 4. Run tests
echo 5. Check status
echo 6. Install dependencies
echo 7. Exit
echo.

:choice
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto gui
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto demo
if "%choice%"=="4" goto test
if "%choice%"=="5" goto status
if "%choice%"=="6" goto install
if "%choice%"=="7" goto exit
echo Invalid choice. Please enter 1-7.
goto choice

:gui
echo.
echo Starting GUI interface with model setup...
echo This will open a full GUI application with model configuration
python detection_gui.py
goto after_run

:cli
echo.
echo Starting CLI interface...
python cli.py --help
echo.
echo Example commands:
echo   python cli.py --list-models
echo   python cli.py --add-model ball --yolo-model tennis_ball.pt
echo   python cli.py --test-image frame.jpg --mode ball
python cli.py
goto after_run

:demo
echo.
echo Running demonstration...
python demo.py
goto after_run

:test
echo.
echo Running tests...
python test_detection_engine.py
goto after_run

:status
echo.
echo Checking engine status...
python cli.py --status
goto after_run

:install
echo.
echo Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo ERROR: requirements.txt not found
)
goto after_run

:after_run
echo.
echo ===============================================
echo Operations completed. Press any key to return to menu...
pause >nul
cls
goto start

:exit
echo.
echo Thank you for using Tennis Trajectory Detection Engine!
pause
exit /b 0

:start
cls
echo ===============================================
echo Tennis Trajectory Detection Engine - YOLOv8n
echo ===============================================
echo.
echo Available options:
echo 1. Run GUI interface (with model setup)
echo 2. Run CLI interface
echo 3. Run demo
echo 4. Run tests
echo 5. Check status
echo 6. Install dependencies
echo 7. Exit
echo.
goto choice