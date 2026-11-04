# Otimização estrutural via aprendizado por reforço e modelos substitutos

Códigos do Trabalho de Conclusão de Curso de Matheus de Moraes Gomes
(Engenharia Mecânica, Escola Politécnica da USP, 2026), orientado pelo
Prof. Dr. William Manjud Maluf Filho.

## Ambiente

Todos os resultados do texto foram gerados em uma estação de trabalho AMD
Ryzen Threadripper PRO 5995WX (128 threads, 512 GB de RAM), em CPU. A única
exceção é o treino da U-Net de tensões (`t02_unet_tensao.py`), que usa GPU
quando há uma disponível; sem GPU ele roda em CPU, só que mais devagar.

- Python 3.9.25
- numpy 1.23.5, scipy 1.13.1, matplotlib 3.9.4, pandas 2.3.3, scikit-learn 1.6.1
- torch 2.8.0, gymnasium 1.1.1, stable-baselines3 2.7.1, sb3-contrib 2.7.1
- tensorflow 2.12.0, tensorboard 2.12.3 e gym 0.26.2 (apenas para o ambiente
  de Brown et al. (2022), usado em `04_benchmark_brown.py`, no painel do DQN de
  `09_resultados_cantilever.py`, em `t01`/`t03` e nos scripts de `tentativa/`;
  todos esperam o código original dos autores em `brown_original/`)

## Execução

Ordem usada para gerar os resultados do texto:

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
# antes de 04, 09, t01, t03 e tentativa/: colocar o código de Brown et al. em brown_original/ (ver abaixo)
python 01_fem_simp.py
python 13_checkerboard.py
python 02_dados_surrogate.py      # coleta do dataset (~3 min)
python 03_surrogates.py           # --so_tempo_mef: mede só a latência do MEF, sem treinar
python 07_testes_ambiente.py
python 08_treinar_ppo.py --problem cantilever --nelx 20 --nely 10 \
    --vol_frac 0.5 --simetria horizontal --steps 1500000
for s in 42 7 13 21 99; do python 08_treinar_ppo.py --problem automotive --seed $s; done
python 08_treinar_ppo.py --problem automotive --dfam_peso 10 --dfam_base baixo
python 09_resultados_cantilever.py
python 10_simp_automotivo.py
python 11_resultados_automotivo.py
python 12_analises_automotivo.py --so tudo
python 14_campanha_dfam.py
python 15_tensao_deslocamento.py

# substituto do campo de tensões (U-Net) no DQN de Brown et al.
python t01_dados_brown.py --workers 96          # dataset com o MEF de Brown (~28 min com 96 processos)
python t02_unet_tensao.py                        # modelo A
python t03_benchmark_brown_unet.py               # figura t03_benchmark_brown_unet_A (metas 0,35/0,45/0,45)

# na pasta tentativa/: escolha das metas, modelo B (hold-out) e diagnóstico
cd tentativa
python metas_escolha.py                          # metas de volume, só com o MEF
python t02_unet_tensao.py --holdout              # modelo B, sem os casos do benchmark
python t03_benchmark_brown_unet.py --sufixo _B --vol_ponte 0.45 --vol_console 0.45 \
    --modelo resultados/dados/unet_tensao_brown_24x24_holdout.pth
```

`04_benchmark_brown.py` foi a primeira tentativa de acoplamento (CNN da
compliância no laço do DQN). Ela não entra mais no texto, que usa a U-Net; o
script fica no repositório como registro. Os `t01`–`t03` desta pasta são cópias
dos de `tentativa/` (que guarda o histórico das rodadas em
`tentativa/LEIA-ME_tentativa.txt`), com os caminhos ajustados e, no `t03`, os
padrões da rodada A.

Tudo o que a execução gera fica em `resultados/` (e em `tentativa/resultados/`).
Esta pasta traz, sem alterações, o de uma execução completa a partir de uma
pasta limpa (02 e 03/11/2026). Nela, 01, 13, 07 e 02 rodaram em sequência, em
paralelo com os sete treinos; depois 03, 04, 09, 10, 11, 12, 14 e 15, um de cada
vez. Os `t01`–`t03` rodaram em 04/11/2026.

- `resultados/*.png` e `*.csv`: figuras e tabelas;
- `resultados/dados/`: dataset e modelos substitutos, topologia final do PPO no
  suporte, a U-Net do modelo A (`unet_tensao_brown_24x24.pth`) e, para cada
  treino do PPO, o melhor modelo (`best_model.zip`, menor compliância na
  avaliação) e o modelo final (`ppo_*_final.zip`). O `08_treinar_ppo.py` apaga os
  checkpoints periódicos ao fim do treino (eles só servem para retomar um treino
  interrompido); `--manter_checkpoints` os preserva. Os arquivos `.pkl` e `.zip`
  são guardados com Git LFS;
- o dataset do `t01` (`brown_tensao_24x24.npz`, 174 MB) não é versionado, por
  estar acima do limite do GitHub; o `t01_dados_brown.py` o regenera;
- `resultados/logs/`: curva de treino de cada execução do `08_treinar_ppo.py`
  (`treino.monitor.csv`: recompensa e comprimento de cada episódio);
- `resultados/console/`: saída de console de cada etapa (`01.log` a `15.log`;
  `08_*.log` para os treinos), de onde saem os números citados no texto, e
  `status.log`, com início, fim e código de saída de cada etapa (o 15 rodou à
  parte, em 03/11, e não consta dele). Os comandos acima não a gravam: ela foi
  obtida redirecionando a saída de cada comando para um arquivo.

Tempo do MEF em `03_surrogates.py`: até 04/11/2026, a função `compliance_fem`
convolvia a densidade com a matriz 576×576 do filtro, e o tempo de 205 ms que
aparece em `resultados/console/03.log` era quase todo dessa conta. Corrigida, a
análise MEF leva 4,8 ms (`resultados/console/03_tempo_mef.log`, medido com
`--so_tempo_mef`, sem retreinar). Os modelos e as figuras do 03 não mudam, pois a
função só mede tempo.

Reprodutibilidade: com os modelos de `resultados/dados/`, as análises (SIMP,
heurística de remoção, robustez, Monte Carlo, *overhang*, malhas e comparações)
reproduzem os CSVs e as figuras desta pasta byte a byte. Os treinos não têm
garantia de reprodução bit a bit. Numa rodada anterior na mesma máquina, os
seis treinos do suporte deram exatamente as mesmas compliâncias; o treino do
cantilever (C = 70,85 contra 71,31) e a CNN de `03_surrogates.py` (R² = 0,9848
contra 0,9891) não. O DQN de Brown com pesos carregados é guloso, então suas
topologias se repetem; já os tempos (figuras de `03_surrogates.py` e
`t03_benchmark_brown_unet.py`) variam a cada execução.

Os scripts que usam o código de Brown et al. (2022) não o distribuem: o
repositório original (<https://github.com/nkbrown503/PhDResearch>) não declara
licença. Coloque o diretório `Top_Opt_RL` do commit `9fcac73` em
`brown_original/` e altere dois arquivos de `Top_Opt_RL/DQN`: em
`RL_Necessities.py`, importar `Adam` de `tensorflow.keras.optimizers.legacy`,
para carregar os pesos originais com TensorFlow 2.11 ou superior; e, em
`TopOpt_Env_Functions.py`, calcular `Final_Results` no início de `Testing_Info` e
deixar de imprimir e plotar a topologia, o que permite chamá-la sem interface
gráfica. Os números de linha citados nos comentários referem-se ao commit
`9fcac73` sem essas duas alterações.

## Scripts e resultados do texto

| Script | O que faz | No texto |
|---|---|---|
| `01_fem_simp.py` | MEF Q4 e SIMP; verificação contra Kharmanda (2004) e caso 20×10 | Seção 4.1 |
| `02_dados_surrogate.py` | coleta 6.000 amostras de iterações intermediárias do SIMP | Seção 3.3 |
| `03_surrogates.py` | treina e compara RF, GP+PCA e CNN; latência do MEF | Seções 3.3 e 4.2 |
| `04_benchmark_brown.py` | primeira tentativa: DQN de Brown et al. com a CNN da compliância | fora do texto |
| `05_ambiente_rl.py` | ambiente do agente: máscara de ações, recompensa, penalidade de overhang | Seção 3.4 |
| `06_politica_fcn.py` | política totalmente convolucional | Seção 3.4 |
| `07_testes_ambiente.py` | testes do ambiente (máscara, cache, bloqueio) | — |
| `08_treinar_ppo.py` | treino do PPO com mascaramento (`--seed`, `--dfam_peso`) | Seções 4.3 e 4.5 |
| `09_resultados_cantilever.py` | curva de treino e comparação SIMP × DQN × PPO no cantilever | Seção 4.3 |
| `10_simp_automotivo.py` | SIMP do suporte e estudo de ablação | Seção 4.4 |
| `11_resultados_automotivo.py` | SIMP × heurística ESO × PPO no suporte, frações de volume e estatística das sementes | Seção 4.5 |
| `12_analises_automotivo.py` | robustez, Monte Carlo multivariado, overhang e malhas refinadas | Seções 4.6 a 4.8 |
| `13_checkerboard.py` | efeito do filtro de sensibilidade (padrão em xadrez) | Seção 3.1 |
| `14_campanha_dfam.py` | agente de referência × agente treinado com penalidade de overhang | Seção 4.7 |
| `15_tensao_deslocamento.py` | verificação a posteriori de tensão e deslocamento no suporte | Seção 4.9 |
| `t01_dados_brown.py` | dataset do substituto de campo com o MEF de Brown | Seção 3.3 |
| `t02_unet_tensao.py` | U-Net do campo de tensões e da energia (modelos A e B) | Seção 3.3 |
| `t03_benchmark_brown_unet.py` | DQN de Brown com e sem a U-Net no laço | Seção 4.2 |
| `tentativa/metas_escolha.py` | metas de volume da ponte e do console, só com o MEF | Seção 4.2 |
| `tentativa/diag_parada.py` | diagnóstico passo a passo da etapa 24×24 | — |

## Resultados de referência

- Verificação (Kharmanda, 32×20): C = 57,3492 em 71 iterações.
- Cantilever 20×10: SIMP contínuo 88,88; SIMP binarizado 66,75; PPO 71,31;
  heurística ESO com simetria 78,90.
- Suporte 30×30: SIMP 22,93 (binarizado 20,72); heurística ESO 20,42; PPO
  (semente 42) 20,32; cinco sementes 20,33 ± 0,03.
- U-Net no DQN de Brown (modelo A): 0 chamadas ao MEF 24×24 nas remoções;
  aceleração de 13,2×, 7,1× e 5,6× (cantilever, ponte, console) com compliância
  1,04, 1,00 e 1,01 vez a do DQN original. Modelo B (sem os casos do
  benchmark no treino): 1,55 no cantilever.
