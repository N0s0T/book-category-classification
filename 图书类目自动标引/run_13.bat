@echo off
echo Running improved model...
py -3.12 13.py > output_13.txt 2>&1
type output_13.txt
echo.
echo Results saved to output_13.txt
pause
