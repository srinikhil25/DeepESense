#!/bin/bash
# Sequential QE runner — one material at a time to avoid OOM on 8GB system
set -euo pipefail

NPROC=8   # 8 of 16 cores; keeps memory under 2GB per material
PSEUDO_DIR="/mnt/d/DeepESense/pseudo/SSSP_1.3.0_PBE_efficiency"
QE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${QE_DIR}/sequential_run.log"

source /home/srig/miniforge3/etc/profile.d/conda.sh
conda activate qe

echo "=== Sequential QE run started at $(date) ===" | tee -a "$LOG"
echo "pw.x: $(which pw.x)" | tee -a "$LOG"

# Materials in order of cell size (smallest first)
MATERIALS=(mp-1207345 mp-1209229 mp-1220948 mp-1182442 mp-640562 mp-622591 mp-1196923 mp-505089)

TOTAL=${#MATERIALS[@]}
IDX=0

for MAT in "${MATERIALS[@]}"; do
    IDX=$((IDX + 1))
    DIR="${QE_DIR}/${MAT}"
    echo "" | tee -a "$LOG"
    echo "[$IDX/$TOTAL] ===== $MAT =====" | tee -a "$LOG"

    if [ ! -f "$DIR/scf.in" ]; then
        echo "  SKIP: no scf.in" | tee -a "$LOG"
        continue
    fi

    cd "$DIR"

    # Inject pseudo_dir if needed
    for inp in scf.in nscf.in; do
        if [ -f "$inp" ] && ! grep -q "pseudo_dir" "$inp"; then
            sed -i "/&CONTROL/a\\  pseudo_dir = '${PSEUDO_DIR}'," "$inp"
        fi
    done

    # Skip if already complete
    if grep -q 'convergence has been achieved' scf.out 2>/dev/null; then
        if grep -q 'JOB DONE' nscf.out 2>/dev/null || grep -q 'highest occupied' nscf.out 2>/dev/null; then
            echo "  ALREADY COMPLETE — skipping" | tee -a "$LOG"
            continue
        fi
    fi

    # SCF
    if grep -q 'convergence has been achieved' scf.out 2>/dev/null; then
        echo "  SCF already converged — skipping to NSCF" | tee -a "$LOG"
    else
        echo "  SCF start: $(date)" | tee -a "$LOG"
        T0=$(date +%s)
        if mpirun -np $NPROC pw.x -in scf.in > scf.out 2>&1; then
            T1=$(date +%s)
            DT=$((T1 - T0))
            if grep -q 'convergence has been achieved' scf.out; then
                echo "  SCF CONVERGED in ${DT}s" | tee -a "$LOG"
            else
                echo "  SCF finished in ${DT}s — NOT converged" | tee -a "$LOG"
                continue
            fi
        else
            T1=$(date +%s)
            DT=$((T1 - T0))
            echo "  SCF FAILED in ${DT}s" | tee -a "$LOG"
            continue
        fi
    fi

    # NSCF
    if [ -f nscf.in ]; then
        echo "  NSCF start: $(date)" | tee -a "$LOG"
        T0=$(date +%s)
        if mpirun -np $NPROC pw.x -in nscf.in > nscf.out 2>&1; then
            T1=$(date +%s)
            DT=$((T1 - T0))
            echo "  NSCF DONE in ${DT}s" | tee -a "$LOG"
            # Extract gap if present
            if grep -q 'highest occupied' nscf.out; then
                grep 'highest occupied' nscf.out | tee -a "$LOG"
            fi
        else
            T1=$(date +%s)
            DT=$((T1 - T0))
            echo "  NSCF FAILED in ${DT}s" | tee -a "$LOG"
        fi
    fi

    # Report memory
    echo "  Memory after: $(free -h | grep Mem | awk '{print $3"/"$2}')" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Sequential run finished at $(date) ===" | tee -a "$LOG"
