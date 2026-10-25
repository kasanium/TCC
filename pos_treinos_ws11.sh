#!/usr/bin/env bash
# roda sozinho depois dos 6 treinos do suporte (run_treinos_ws11.sh):
# analises do suporte, campanha de overhang e regeracao das figuras com
# virgula decimal. cada etapa tem seu log em logs/pos_*.log.
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs
ST=logs/pos_status.log

until grep -q "TODOS OS TREINOS CONCLUIDOS" logs/treinos_ws11.log 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] treinos concluidos, iniciando pos-processamento" >> $ST

etapa() {
    local nome=$1; shift
    echo "[$(date '+%F %T')] inicio $nome" >> $ST
    "$@" > "logs/pos_$nome.log" 2>&1
    echo "[$(date '+%F %T')] fim $nome (exit $?)" >> $ST
}

# suporte: comparacao + fracoes de volume + estatistica das 5 sementes;
# grava a topologia do ppo usada por 12 e 14
etapa 11 python -u 11_resultados_automotivo.py
# simp baseline/ablacao com a cpu livre (tempos limpos)
etapa 10 python -u 10_simp_automotivo.py
# robustez, monte carlo multivariado, dfam e malha (tempos limpos)
etapa 12 python -u 12_analises_automotivo.py --so tudo
# campanha com penalidade de overhang
etapa 14 python -u 14_campanha_dfam.py

# figuras do cantilever e do benchmark do brown com virgula decimal
etapa 09 python -u 09_resultados_cantilever.py
etapa 04 python -u 04_benchmark_brown.py

# substitutos: copia isolada, para nao sobrescrever os modelos usados pelo 04
R03=~/regen03_$(date +%Y%m%d)
mkdir -p $R03/resultados/dados
cp 01_fem_simp.py 03_surrogates.py $R03/
cp resultados/dados/topo_data_24x24.npz $R03/resultados/dados/
etapa 03 bash -c "cd $R03 && python -u 03_surrogates.py"

echo "[$(date '+%F %T')] POS-PROCESSAMENTO CONCLUIDO" >> $ST
