# Ghost Busters — Process Monitor Utilities

Debugging tools for monitoring and killing runaway processes during long-running SLURM experiments.

## Contents

| File | Description |
|------|-------------|
| `cpu_killer.py` | Monitors a process by PID and kills it if CPU or memory usage exceeds configured thresholds |
| `ghost_hunting.ipynb` | Interactive notebook for investigating zombie/orphan processes on shared compute nodes |

## Usage

```bash
# Monitor PID 12345, kill if CPU > 1500% or memory > 10 GB
python cpu_killer.py <PID>
```

### Configuration (in `cpu_killer.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CPU_LIMIT` | 1500 (%) | Kill threshold (e.g., 1500% = 15 cores) |
| `MEM_LIMIT_MB` | 10000 | Memory threshold in MB |
| `REFRESH_RATE` | 1 | Seconds between checks |

## Context

These utilities were created to handle edge cases on PLGrid/Athena HPC where model processes occasionally hang or leak memory after SLURM timeouts, preventing subsequent array tasks from starting.
