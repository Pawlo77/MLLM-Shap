"""
CPU & Memory Killer Monitor
"""

import time
import sys
import psutil

CPU_LIMIT: int = 1500
"""Cpu limit in percentage (e.g., 1500% for 15 cores)"""

MEM_LIMIT_MB: int = 10000
"""Memory limit in megabytes"""

REFRESH_RATE: int = 1
"""Seconds between checks"""


def get_process_stats(
    pid: int,
) -> tuple[psutil.Process | None, str | None, float, float]:
    """Get CPU and Memory stats for a process by PID."""
    try:
        p = psutil.Process(pid)
        mem_mb = p.memory_info().rss / 1024 / 1024
        c = p.cpu_percent(interval=0.1)
        return p, p.name(), c, mem_mb
    except psutil.NoSuchProcess:
        return None, None, 0, 0


# pylint: disable=too-many-statements,too-many-branches
def main() -> None:
    """Main monitoring loop."""
    if len(sys.argv) < 2:  # pylint: disable=magic-value-comparison
        print("Usage: python monitor_mem.py <PID>")
        sys.exit(1)

    target_pid = int(sys.argv[1])
    print(f"Watching PID {target_pid}...")

    max_mem_seen: float = 0
    min_mem_seen: float = float("inf")
    while True:
        try:
            # 1. Get Stats
            proc_obj, name, cpu, mem = get_process_stats(target_pid)

            if name is None:
                print(f"\nProcess {target_pid} finished or does not exist.")
                break

            max_mem_seen = max(max_mem_seen, mem)
            min_mem_seen = min(min_mem_seen, mem)

            # 2. Draw Interface
            print("\033[H\033[J", end="")  # Clear screen
            print("==========================================")
            print(f"   MONITOR: {name} (PID: {target_pid})")
            print("==========================================")
            print(f" REAL RAM : {mem:.2f} MB  (Limit: {MEM_LIMIT_MB} MB)")
            print(f" MINI RAM : {min_mem_seen:.2f} MB")
            print(f" PEAK RAM : {max_mem_seen:.2f} MB")
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

            if choice == "k":  # pylint: disable=magic-value-comparison
                if proc_obj and proc_obj.is_running():
                    print(f"Killing PID {target_pid}...")
                    proc_obj.kill()
                    print("Process terminated.")
                else:
                    print("Process is already gone.")
                sys.exit(0)

            elif choice == "q":  # pylint: disable=magic-value-comparison
                print("Exiting monitor.")
                sys.exit(0)

            else:
                print("Resuming...")
                time.sleep(0.5)
                continue


if __name__ == "__main__":
    main()
