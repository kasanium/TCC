"""diagnostico: por que o dqn de brown para na etapa 24x24.

roda um caso do benchmark (com o mef e/ou com a u-net) e imprime, para cada
passo da malha 24x24, a acao, se o elemento era solido, se e apoio/carga, se a
remocao mantem o corpo conexo e as grandezas da l. 60 de TopOpt_Env_Functions.py
(razao de energia Max_SE_Tot/Max_SE_Ep e contagem It); no ultimo passo diz qual
condicao encerrou o episodio.
uso: python diag_parada.py [caso] [mef|unet|ambos] [vol]
     ex.: python diag_parada.py 2_Ponte mef
"""
import os
import sys
import math
from importlib.util import spec_from_file_location, module_from_spec

CASO = sys.argv[1] if len(sys.argv) > 1 else "1_Cantilever"
MODO = sys.argv[2] if len(sys.argv) > 2 else "ambos"
VOL = float(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = sys.argv[:1]   # o t03 le os proprios argumentos na importacao

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("t03", os.path.join(_DIR, "t03_benchmark_brown_unet.py"))
t03 = module_from_spec(_spec)
_spec.loader.exec_module(t03)

import numpy as np
from unittest.mock import patch
from Top_Opt_RL.DQN import TopOpt_Env_Functions as TEF
from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import isolate_largest_group_original

step_orig = TEF.TopOpt_Gen.step


def step_log(self, action, observation, Last_Reward, load_checkpoint, env, PR, FEA_Skip):
    if self.EX * self.EY != 576:
        return step_orig(self, action, observation, Last_Reward, load_checkpoint, env, PR, FEA_Skip)
    a = int(action)
    solido = self.VoidCheck[a] == 1
    bc = a in [int(b) for b in np.asarray(self.BC).ravel()]
    teste = np.array(self.VoidCheck, dtype=float).copy()
    teste[a] = 0
    conexo = isolate_largest_group_original(np.reshape(teste, (self.EX, self.EY)))[1]
    obs = np.asarray(observation)[:, :, 0].ravel()
    rank = int((obs > obs[a]).sum())
    out = step_orig(self, action, observation, Last_Reward, load_checkpoint, env, PR, FEA_Skip)
    it = list(self.VoidCheck).count(0)
    tot = float(np.max(self.Max_SE_Tot))
    ep = float(self.Max_SE_Ep)
    razao = abs(tot / ep) if ep else float("nan")
    print(f"    passo {self.Counter:3d} | acao {a:3d} | solido {solido} | apoio/carga {bc} | "
          f"conexo {conexo} | obs[a] {obs[a]:.3f} (posto {rank}) | It {it} | "
          f"SE_tot {tot:.4g} | SE_ep {ep:.4g} | razao {razao:.3f} | done {out[2]}")
    if out[2]:
        meta = math.ceil(self.EX * self.EY * (1 - self.Vol_Frac))
        motivos = []
        if not (solido and not bc and conexo):
            motivos.append("acao ilegal (" + ", ".join(
                m for m, v in [("elemento ja vazio", not solido), ("apoio/carga", bc),
                               ("desconectaria o corpo", not conexo)] if v) + ")")
        else:
            if it >= meta:
                motivos.append(f"meta de volume atingida (It {it} >= {meta})")
            if razao >= 1:
                motivos.append(f"razao de energia >= 1 ({razao:.3f})")
            if it / (self.EX * self.EY) >= 1:
                motivos.append("It = numero de elementos")
            if tot == 0 or ep == 0:
                motivos.append("Max_SE == 0")
        print(f"    >>> fim do episodio: {'; '.join(motivos) or 'restricao de tensao'}")
    return out


conf = dict(t03.TEST_CASES[CASO])
if VOL is not None:
    conf["vol"] = VOL
modos = {"mef": [False], "unet": [True], "ambos": [False, True]}[MODO]
for usar in modos:
    print(f"\n=== {CASO} (vol {conf['vol']}), {'U-Net' if usar else 'MEF'} ===")
    with patch.object(TEF.TopOpt_Gen, "step", step_log):
        t03.run_brown(conf, use_surrogate=usar)
