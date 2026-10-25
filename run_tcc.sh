#!/usr/bin/env bash
# pipeline completo do tcc, na ordem em que os resultados do texto foram
# gerados (estacao de trabalho ws11, cpu). sem argumento roda so as etapas
# rapidas, que reaproveitam dados e checkpoints ja salvos em resultados/dados;
# com --tudo refaz tambem a coleta do dataset, os substitutos, o benchmark do
# brown e todos os treinamentos. para os 6 treinos do suporte em paralelo,
# usar run_treinos_ws11.sh (o pos-processamento automatico e pos_treinos_ws11.sh).
set -e
cd "$(dirname "$0")"
TUDO=0
[ "$1" = "--tudo" ] && TUDO=1

echo "== 01: verificacao fem/simp (kharmanda 57.3492; caso 20x10 88.8854) =="
python3 01_fem_simp.py
echo "== 13: efeito do filtro (checkerboard) =="
python3 13_checkerboard.py

if [ $TUDO -eq 1 ]; then
  echo "== 02: coleta do dataset dos substitutos (~horas) =="
  python3 02_dados_surrogate.py
  echo "== 03: treino dos substitutos =="
  python3 03_surrogates.py
  echo "== 04: benchmark dqn de brown + cnn (requer tensorflow) =="
  python3 04_benchmark_brown.py
fi

echo "== 07: testes do ambiente =="
python3 07_testes_ambiente.py

if [ $TUDO -eq 1 ]; then
  echo "== 08: treino ppo cantilever (1.5M passos) =="
  python3 08_treinar_ppo.py --problem cantilever --nelx 20 --nely 10 \
      --vol_frac 0.5 --simetria horizontal --steps 1500000 --n_envs 32
  echo "== 08: treino ppo suporte, 5 sementes (3M passos cada) =="
  for s in 42 7 13 21 99; do
    python3 08_treinar_ppo.py --problem automotive --seed $s
  done
  echo "== 08: treino ppo suporte com penalidade de overhang (w=10, base embaixo) =="
  python3 08_treinar_ppo.py --problem automotive --dfam_peso 10 --dfam_base baixo
fi

echo "== 09: avaliacao cantilever (curvas, brown) =="
python3 09_resultados_cantilever.py
echo "== 10: simp suporte (baseline 22.93 + ablacao) =="
python3 10_simp_automotivo.py
echo "== 11: comparacao suporte (simp vs eso vs ppo) + estatistica das sementes =="
python3 11_resultados_automotivo.py
echo "== 12: robustez + monte carlo multivariado + dfam + malha =="
python3 12_analises_automotivo.py --so tudo
echo "== 14: campanha com penalidade de overhang =="
python3 14_campanha_dfam.py
echo "pipeline concluido; figuras em resultados/"
