@echo off
chcp 65001 >nul
python run_svm_tune.py > svm_output.txt 2>&1
type svm_output.txt