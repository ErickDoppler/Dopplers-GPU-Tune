@echo off
rem Dopplers GPU Tune - build with the w64devkit toolchain (c:\workenv\w64devkit).
rem   build.bat        -> GpuTune.exe      (asks for Administrator; changes the VRAM clock)
rem   build.bat dry    -> also GpuTune-dry.exe (never writes an offset, no elevation: for testing the window and the measurement)
set PATH=c:\workenv\w64devkit\bin;%PATH%
cd /d %~dp0
set LIBS=-lcomctl32 -lgdi32 -luser32 -lshell32 -lwinhttp -lcomdlg32
windres -I res -i res\admin.rc -o res\admin.res.o || exit /b 1
g++ -std=c++17 -O2 -s -static -mwindows -municode -Iinclude src\gputune.cpp res\admin.res.o %LIBS% -o GpuTune.exe || exit /b 1
echo built GpuTune.exe
if /i "%1"=="dry" (
  windres -I res -i res\user.rc -o res\user.res.o || exit /b 1
  g++ -std=c++17 -O2 -s -static -mwindows -municode -DDRY_RUN -Iinclude src\gputune.cpp res\user.res.o %LIBS% -o GpuTune-dry.exe || exit /b 1
  echo built GpuTune-dry.exe
)
