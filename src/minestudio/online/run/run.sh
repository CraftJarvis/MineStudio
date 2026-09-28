ps  -elf | grep 'multiprocess' | awk '{print $4}' | xargs kill -9 
ps  -elf | grep 'ray' | awk '{print $4}' | xargs kill -9 

LOG_FILE="output/memory_log.txt"  # 日志文件路径，可以根据需要修改
> "$LOG_FILE"
# 确保日志文件存在
touch "$LOG_FILE"
echo "Memory monitor started. Logging to: $LOG_FILE"
while true; do
    # 获取内存使用率
    #echo "Checking memory usage..." >> "$LOG_FILE"
    mem_usage=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    if (( $(echo "$mem_usage > 90" | bc -l) )); then
        echo "-------------------------" >> "$LOG_FILE"
        echo "$(date): High memory usage detected: ${mem_usage}%" >> "$LOG_FILE"
        echo "Current memory usage by users:" >> "$LOG_FILE"
        ps -eo user,pid,%mem,command --sort=-%mem | head -n 100 >> "$LOG_FILE"
        echo "Killing 'multiprocess' and 'ray' processes..." >> "$LOG_FILE"
        
        # 终止 'multiprocess' 相关进程
        ps -elf | grep 'multiprocess' | grep -v grep | awk '{print $4}' | while read pid; do
            kill -9 "$pid" && echo "Killed multiprocess PID: $pid" >> "$LOG_FILE"
        done
        
        # 终止 'ray' 相关进程
        ps -elf | grep 'ray' | grep -v grep | awk '{print $4}' | while read pid; do
            kill -9 "$pid" && echo "Killed ray PID: $pid" >> "$LOG_FILE"
        done

        echo "Exiting memory monitor due to high usage." >> "$LOG_FILE"
        exit 1
    fi

    sleep 5  # 每5秒检查一次
done &

Xvfb :4 -maxclients 1024 &
export DISPLAY=:4
# export DISPLAY=":1"
# Xvfb "${DISPLAY}" -ac -screen "0" "1920x1200x24" -dpi "72" +extension "RANDR" +extension "GLX" +iglx +extension "MIT-SHM" +render -nolisten "tcp" -noreset -shmem -maxclients 2048 &
bash start_headnode.sh
python run.py
