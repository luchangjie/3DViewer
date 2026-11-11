@echo off
REM ============================================================
REM 3DViewer 一键构建脚本  (Release | x64)
REM 依赖：Visual Studio 2022 + CUDA Toolkit
REM ============================================================
setlocal enabledelayedexpansion

set "SLN=3DViewer.sln"

REM 用 vswhere 定位 VS 安装路径，再取其 x64 开发者命令环境
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="

if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if not exist "%VCVARS%" (
  echo [错误] 找不到 vcvars64.bat。
  echo 请确认已安装 Visual Studio 2022（含“使用 C++ 的桌面开发”），
  echo 或自行打开 “x64 Native Tools Command Prompt for VS” 后运行：
  echo   msbuild %SLN% /p:Configuration=Release /p:Platform=x64
  exit /b 1
)

call "%VCVARS%"
msbuild %SLN% /p:Configuration=Release /p:Platform=x64
if errorlevel 1 (
  echo [错误] 构建失败，请查看上面的输出。
  exit /b 1
)

echo.
echo [完成] 产物位于 x64\Release\  （3DViewer.exe + render_kernel.dll）
echo         两个文件必须放在同一目录运行。
endlocal
