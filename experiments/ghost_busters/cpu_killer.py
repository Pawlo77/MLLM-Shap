"""
CPU & Memory Killer Monitor
"""

import time
import sys
import psutil

# --- CONFIGURATION ---
CPU_LIMIT = 1500       # % CPU limit
MEM_LIMIT_MB = 10000   # MB Memory limit
REFRESH_RATE = 1       # Seconds between checks


def get_process_stats(pid: int) -> tuple[psutil.Process | None, str | None, float, float]:
    """Get CPU and Memory stats for a process by PID."""
    try:
        p = psutil.Process(pid)
        mem_mb = p.memory_info().rss / 1024 / 1024
        c = p.cpu_percent(interval=0.1)
        return p, p.name(), c, mem_mb
    except psutil.NoSuchProcess:
        return None, None, 0, 0


if len(sys.argv) < 2:  # pylint: disable=magic-value-comparison
    print("Usage: python monitor_mem.py <PID>")
    sys.exit(1)

target_pid = int(sys.argv[1])
print(f"Watching PID {target_pid}...")

MAX_MEM_SEEN: float = 0
MIN_MEM_SEEN: float = float('inf')

while True:
    try:
        # 1. Get Stats
        proc_obj, name, cpu, mem = get_process_stats(target_pid)

        if name is None:
            print(f"\nProcess {target_pid} finished or does not exist.")
            break

        MAX_MEM_SEEN = max(MAX_MEM_SEEN, mem)
        MIN_MEM_SEEN = min(MIN_MEM_SEEN, mem)

        # 2. Draw Interface
        print("\033[H\033[J", end="")  # Clear screen
        print("==========================================")
        print(f"   MONITOR: {name} (PID: {target_pid})")
        print("==========================================")
        print(f" REAL RAM : {mem:.2f} MB  (Limit: {MEM_LIMIT_MB} MB)")
        print(f" MINI RAM : {MIN_MEM_SEEN:.2f} MB")
        print(f" PEAK RAM : {MAX_MEM_SEEN:.2f} MB")
        print(f" CPU USE  : {cpu:.1f} %    (Limit: {CPU_LIMIT}%)")
        print("==========================================")
        print(" [Press Ctrl+C for Menu: Kill / Quit] ")

        # 3. Auto-Kill Logic
        if mem > MEM_LIMIT_MB:
            print(f"\n[!!!] MEMORY LEAK DETECTED: {mem:.2f} MB")
            if proc_obj is not None:
                proc_obj.kill()
            print("Killed.")
            break

        if cpu > CPU_LIMIT:
            print(f"\n[!!!] CPU SPIKE DETECTED: {cpu}%")
            if proc_obj is not None:
                proc_obj.kill()
            print("Killed.")
            break

        time.sleep(REFRESH_RATE)

    # 4. INTERACTIVE MENU (Catch Ctrl+C)
    except KeyboardInterrupt:
        print("\n\n--- PAUSED ---")
        print("Select option:")
        print("  [K] Kill Process immediately")
        print("  [Q] Quit Monitor (Process keeps running)")
        print("  [Enter] Resume Monitoring")

        choice = input(">> ").lower().strip()

        if choice == 'k':  # pylint: disable=magic-value-comparison
            if proc_obj and proc_obj.is_running():
                print(f"Killing PID {target_pid}...")
                proc_obj.kill()
                print("Process terminated.")
            else:
                print("Process is already gone.")
            sys.exit(0)

        elif choice == 'q':  # pylint: disable=magic-value-comparison
            print("Exiting monitor.")
            sys.exit(0)

        else:
            print("Resuming...")
            time.sleep(0.5)
            continue
