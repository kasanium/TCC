# Otimização estrutural via aprendizado por reforço e modelos substitutos

Códigos do TCC de Matheus de Moraes Gomes (Engenharia Mecânica, Escola
Politécnica da USP, 2026), orientado pelo Prof. Dr. William Manjud Maluf Filho.

## Ambiente

AMD Ryzen Threadripper PRO 5995WX (128 threads, 512 GB de RAM), em CPU; o
treino do `04b` usou GPU. Python 3.9.25, versões em `requirements.txt`.

`04`, `04a`, `04c` e `09` usam o código de Brown et al. (2022)
(<https://github.com/nkbrown503/PhDResearch>, sem licença declarada, por isso
não incluído). Copie `Top_Opt_RL` do commit `9fcac73` para `brown_original/` e,
em `Top_Opt_RL/DQN`: importe `Adam` de `tensorflow.keras.optimizers.legacy` em
`RL_Necessities.py`; em `TopOpt_Env_Functions.py`, calcule `Final_Results` no
início de `Testing_Info` e retire a impressão e o gráfico da topologia.

## Execução

```bash
pip install -r requirements.txt
python 01_fem_simp.py
python 13_checkerboard.py
python 02_dados_surrogate.py
python 03_surrogates.py
python 04_benchmark_brown.py
python 07_testes_ambiente.py
python 08_treinar_ppo.py --problem cantilever --nelx 20 --nely 10 --vol_frac 0.5 \
    --simetria horizontal --steps 1500000
for s in 42 7 13 21 99; do python 08_treinar_ppo.py --problem automotive --seed $s; done
python 08_treinar_ppo.py --problem automotive --dfam_peso 10 --dfam_base baixo
python 09_resultados_cantilever.py
python 10_simp_automotivo.py
python 11_resultados_automotivo.py
python 12_analises_automotivo.py --so tudo
python 14_campanha_dfam.py
python 15_tensao_deslocamento.py
python 04a_dados_brown.py --workers 96
python 04b_unet_tensao.py              # modelo A
python 04b_unet_tensao.py --holdout    # modelo B, sem os casos do benchmark
python 04c_benchmark_brown_unet.py     # modelo A
python 04c_benchmark_brown_unet.py --sufixo _B \
    --modelo resultados/dados/unet_tensao_brown_24x24_holdout.pth
```

## Saídas

- `resultados/`: figuras e tabelas.
- `resultados/dados/`: datasets, modelos substitutos, U-Net e modelos do PPO
  (`.pkl` e `.zip` em Git LFS). O dataset do `04a` (174 MB) não está no
  repositório.
- `resultados/logs/`: curvas de treino do PPO.
- `resultados/console/`: saída de cada script, de onde vêm os números do texto.
  O tempo do MEF em `03.log` (205 ms) está errado; o valor certo (4,8 ms) está
  em `03_tempo_mef.log` (`python 03_surrogates.py --so_tempo_mef`).

Os treinos (PPO, CNN, U-Net) podem não se repetir bit a bit; os tempos medidos
variam a cada execução.

## Scripts

| Script | O que faz | Texto |
|---|---|---|
| `01_fem_simp.py` | MEF Q4 e SIMP; verificação contra Kharmanda (2004) | 4.1 |
| `02_dados_surrogate.py` | dataset dos substitutos (iterações do SIMP) | 3.3 |
| `03_surrogates.py` | RF, GP+PCA e CNN; latência do MEF | 3.3, 4.2 |
| `04_benchmark_brown.py` | CNN no laço do DQN de Brown (primeira tentativa) | 4.2 |
| `04a_dados_brown.py` | dataset da U-Net com o MEF de Brown | 3.3 |
| `04b_unet_tensao.py` | U-Net do campo de tensão e da energia | 3.3 |
| `04c_benchmark_brown_unet.py` | DQN de Brown com e sem a U-Net | 4.2 |
| `05_ambiente_rl.py` | ambiente do agente: máscara, recompensa, overhang | 3.4 |
| `06_politica_fcn.py` | política totalmente convolucional | 3.4 |
| `07_testes_ambiente.py` | testes do ambiente | — |
| `08_treinar_ppo.py` | treino do PPO | 4.3, 4.5 |
| `09_resultados_cantilever.py` | cantilever: SIMP × DQN × PPO | 4.3 |
| `10_simp_automotivo.py` | SIMP do suporte e ablação | 4.4 |
| `11_resultados_automotivo.py` | suporte: SIMP × ESO × PPO | 4.5 |
| `12_analises_automotivo.py` | robustez, Monte Carlo, overhang, malhas | 4.6 a 4.8 |
| `13_checkerboard.py` | padrão em xadrez e filtro | 3.1 |
| `14_campanha_dfam.py` | treino com penalidade de overhang | 4.7 |
| `15_tensao_deslocamento.py` | tensão e deslocamento no suporte | 4.9 |
