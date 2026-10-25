#!/usr/bin/env bash
# treinos do suporte com a geometria corrigida, em paralelo no ws11.
# 6 treinos x (32 ambientes + 1 processo principal); 16 threads de blas/torch
# por processo para nao sobrecarregar os 128 nucleos.
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

rodar() {
    local nome=$1; shift
    echo "[$(date '+%F %T')] inicio $nome" >> logs/treinos_ws11.log
    python -u 08_treinar_ppo.py --problem automotive "$@" > "logs/treino_$nome.log" 2>&1
    echo "[$(date '+%F %T')] fim $nome (exit $?)" >> logs/treinos_ws11.log
}

rodar seed42 &
rodar dfam10 --dfam_peso 10 --dfam_base baixo &
for s in 7 13 21 99; do
    rodar seed$s --seed $s &
done
wait
echo "[$(date '+%F %T')] TODOS OS TREINOS CONCLUIDOS" >> logs/treinos_ws11.log
