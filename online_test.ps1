# 无限循环运行 Python 服务
while ($true) {
    Write-Host "Starting server at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')..."
    
    # 启动 Python 脚本
    python ./main.py --test --online_test

    Write-Host "Server exited at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Restarting in 3 seconds..."
    
    # 等待 3 秒再重启
    Start-Sleep -Seconds 3
}
