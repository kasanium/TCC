# Otimização estrutural via aprendizado por reforço e modelos substitutos

Códigos do Trabalho de Conclusão de Curso de Matheus de Moraes Gomes
(Engenharia Mecânica, Escola Politécnica da USP, 2026), orientado pelo
Prof. Dr. William Manjud Maluf Filho.

## Ambiente

Todos os resultados do texto foram gerados em uma estação de trabalho AMD
Ryzen Threadripper PRO 5995WX (128 threads, 512 GB de RAM), somente em CPU.

- Python 3.9.25
- numpy 1.23.5, scipy 1.13.1, matplotlib 3.9.4, pandas 2.3.3, scikit-learn 1.6.1
- torch 2.8.0, gymnasium 1.1.1, stable-baselines3 2.7.1, sb3-contrib 2.7.1
- tensorflow 2.12.0, tensorboard 2.12.3 e gym 0.26.2 (apenas para o ambiente
  de Brown et al. (2022) usado em `04_benchmark_brown.py`, que espera o código
  original dos autores em `brown_original/`)

## Execução

```bash
bash run_tcc.sh          # etapas rápidas, reaproveitando dados e modelos salvos
bash run_tcc.sh --tudo   # refaz também dataset, substitutos, benchmark e treinos
```

Para os seis treinos do suporte em paralelo (cinco sementes e a campanha com
penalidade de overhang), use `run_treinos_ws11.sh`; `pos_treinos_ws11.sh`
espera o fim dos treinos e roda as análises na ordem certa. As saídas ficam em
`resultados/` (figuras e tabelas) e `resultados/dados/` (dados e modelos).

## Scripts e resultados do texto

| Script | O que faz | No texto |
|---|---|---|
| `01_fem_simp.py` | MEF Q4 e SIMP; verificação contra Kharmanda (2004) e caso 20×10 | Figuras 9 e 10 |
| `02_dados_surrogate.py` | coleta 6.000 amostras de iterações intermediárias do SIMP | Seção 3.4 |
| `03_surrogates.py` | treina e compara RF, GP+PCA e CNN | Figura 11, Seção 4.2 |
| `04_benchmark_brown.py` | DQN de Brown et al. com e sem a CNN substituta | Figura 12 |
| `05_ambiente_rl.py` | ambiente do agente: máscara de ações, recompensa, penalidade de overhang | Seção 3.5 |
| `06_politica_fcn.py` | política totalmente convolucional | Seção 3.5 |
| `07_testes_ambiente.py` | testes do ambiente (máscara, cache, bloqueio) | — |
| `08_treinar_ppo.py` | treino do PPO com mascaramento (`--seed`, `--dfam_peso`) | Seções 4.3 e 4.5 |
| `09_resultados_cantilever.py` | curva de treino e comparação SIMP × DQN × PPO no cantilever | Figuras 13 e 14, Tabela 3 |
| `10_simp_automotivo.py` | SIMP do suporte e estudo de ablação | Figuras 15 e 16 |
| `11_resultados_automotivo.py` | SIMP × heurística ESO × PPO no suporte, frações de volume e estatística das sementes | Figura 17, Seção 4.5 |
| `12_analises_automotivo.py` | robustez (A1–A3), Monte Carlo multivariado (A4), overhang e malhas refinadas | Figuras 18–20 e 22, Tabelas 4–7 |
| `13_checkerboard.py` | efeito do filtro de sensibilidade (padrão em xadrez) | Figura 6 |
| `14_campanha_dfam.py` | agente de referência × agente treinado com penalidade de overhang | Figura 21 |

## Resultados de referência

- Verificação (Kharmanda, 32×20): C = 57,3492.
- Cantilever 20×10: SIMP binarizado 66,75; PPO 70,85; heurística ESO 78,90.
- Suporte 30×30: SIMP binarizado 20,72; heurística ESO 20,42; PPO (semente 42)
  20,32; cinco sementes 20,33 ± 0,03.
