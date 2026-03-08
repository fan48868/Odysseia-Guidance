@echo off
setlocal enabledelayedexpansion
title 类脑娘 Docker 管理终端

:: 检查 docker 是否存在
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Docker 环境，请先安装 Docker！
    pause
    exit /b
)

:loop
cls
echo ============================================
echo          类脑娘 Docker 部署管理工具
echo ============================================
echo.
echo   [1] 全量编译并启动 (up -d)
echo   [2] 直接后台启动 (up -d)
echo   [3] 重启机器人服务 (restart bot_app)
echo   [4] 停止并删除容器 (down)
echo   [5] 退出脚本
echo.
echo   NOTE: After editing .env, prefer [3] restart bot_app (usually no full build).
echo   NOTE: Use [1] only when Dockerfile or dependencies changed.
echo.
echo ============================================
set "choice="
set /p choice="请输入数字并按回车: "

:: 简单的输入校验
if "%choice%"=="1" goto op1
if "%choice%"=="2" goto op2
if "%choice%"=="3" goto op3
if "%choice%"=="4" goto op4
if "%choice%"=="5" exit

:: 无效输入处理
echo.
echo [错误] 无效选择: "%choice%"，请输入数字 1 到 5。
pause
goto loop

:: ============================================
:: 功能执行区 - 使用标签隔离，彻底解决括号解析问题
:: ============================================

:op1
echo.
echo [状态] 正在全量编译并启动所有容器...
docker compose up -d --build
goto end_action

:op2
echo.
echo [状态] 正在后台启动容器...
docker compose up -d
goto end_action

:op3
echo.
echo [状态] 正在重启机器人服务 (bot_app)...
echo [NOTE] If you just edited .env, this option is usually enough.
REM 在这里写注释就绝对安全了
REM 这里的 bot_app 对应 docker-compose.yml 里的服务名
docker compose restart bot_app
goto end_action

:op4
echo.
echo [状态] 正在停止运行并删除所有容器...
docker compose down
goto end_action

:end_action
echo.
echo ============================================
if %errorlevel% equ 0 (
    echo [成功] 命令执行完毕！
) else (
    echo [警告] 命令执行似乎遇到了错误，请检查上方日志。
)
echo 按任意键返回主菜单...
pause >nul
goto loop