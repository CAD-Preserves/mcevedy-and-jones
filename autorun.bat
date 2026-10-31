@echo off
setlocal
pushd "%~dp0"

python core\main.py %*
set EXIT_CODE=%ERRORLEVEL%

popd
exit /b %EXIT_CODE%

