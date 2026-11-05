"""
gera o banco de dados do substituto de campo com o proprio mef de brown 22: roda o dqn original (refinamento progressivo 6 -> 12 -> 24) e grava
toda chamada de FEASolve na malha 24x24 (topologia binaria, apoios, cargas,
tensao de von mises por elemento, observacao normalizada e energia).
para diversificar as topologias sem sair da distribuicao que o agente visita,
cada episodio troca a acao do agente por uma acao legal sorteada com
probabilidade eps (eps = 0 reproduz a trajetoria de referencia do 04).
casos: os tres do benchmark (cantilever, ponte, console) e casos de apoio e
carga sorteados. requer brown_original/ (ver README.md) e tensorflow.
uso: python 04a_dados_brown.py [--rapido] [--workers N]
saida: resultados/dados/brown_tensao_24x24.npz
"""

import os
import sys
import time
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.util import spec_from_file_location, module_from_spec

# os checkpoints de brown foram salvos com o otimizador legado do keras; em
# tf >= 2.11 o adam padrao mudou, entao forcamos o legado antes de carregar o modelo
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam

_DIR = os.path.dirname(os.path.abspath(__file__))
_BROWN = os.path.join(_DIR, "brown_original")
sys.path.insert(0, _BROWN)
sys.path.insert(0, _DIR)

import numpy as np
from unittest.mock import patch

# o formato de entrada vive em 04b_unet_tensao.py, carregado pelo caminho do arquivo
_spec = spec_from_file_location("unet_tensao", os.path.join(_DIR, "04b_unet_tensao.py"))
_unet = module_from_spec(_spec)
_spec.loader.exec_module(_unet)
mapas_nos = _unet.mapas_nos

from Top_Opt_RL.DQN.opts import parse_opts
from Top_Opt_RL.DQN.main import EnviromentsRL, TopOpt_Designing
from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import FEASolve, isolate_largest_group_original
from Top_Opt_RL.DQN.RL_Necessities import Agent


# ---------------------------------------------------------------------------
# parametros
# ---------------------------------------------------------------------------

# casos de teste no formato do ambiente de brown 22, mesmos do 04
TEST_CASES = {
    "1_Cantilever": {"bcs": [1, 553],             "downs": [312],        "vol": 0.35},
    "2_Ponte":      {"bcs": [552, 553, 574, 575], "downs": [11, 12, 13], "vol": 0.40},
    "3_Console":    {"bcs": [1, 289, 553],        "downs": [190],        "vol": 0.50},
}

# padroes usados no resultado do texto: 48 / 600 / 3 -> 1.944 episodios
N_EPIS_CASO_TESTE = 48        # por caso do benchmark: 1 de referencia (eps = 0) + 47 perturbados
N_CASOS_ALEATORIOS = 600      # combinacoes de apoio/carga sorteadas
N_EPIS_CASO_ALEATORIO = 3     # por caso sorteado: eps = 0 e eps sorteados
EPS_MIN, EPS_MAX = 0.02, 0.30 # faixa da perturbacao: pouca o bastante para o agente
                              # ainda conduzir o episodio, muita o bastante para variar
VOL_ALEATORIO = (0.25, 0.45)  # vol final dos casos sorteados (0.5 encerra a etapa 24x24 de cara)
SEED = 42

EX = EY = 24

# arestas da malha 24x24 no indice 1-based do app de brown: elemento e (0-based)
# fica em x = e % 24, y = e // 24 (rectangularmesh: d = nx*j + i)
_BORDA_INF = [x + 1 for x in range(EX)]
_BORDA_SUP = [EX * (EY - 1) + x + 1 for x in range(EX)]
_BORDA_ESQ = [EX * y + 1 for y in range(EY)]
_BORDA_DIR = [EX * y + EX for y in range(EY)]


# ---------------------------------------------------------------------------
# gravacao das chamadas ao mef
# ---------------------------------------------------------------------------

GRAVADOR = {"ativo": False, "amostras": [], "vistos": set()}


def fea_gravador(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=True):
    """chama o FEASolve real e grava a chamada se for na malha 24x24."""
    res = FEASolve(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=Stress)
    if GRAVADOR["ativo"] and ex * ey == EX * EY and Stress:
        solido = (np.asarray(void_check, dtype=np.float64).ravel() > 0.5).astype(np.uint8)
        mapas = mapas_nos(lc_nodes, load_dirs, bc_nodes, ex, ey)
        chave = solido.tobytes() + mapas.tobytes()
        if chave not in GRAVADOR["vistos"]:   # step e Testing_Info repetem o mesmo estado
            GRAVADOR["vistos"].add(chave)
            GRAVADOR["amostras"].append({
                "solido": solido,
                "mapas": mapas,
                "vm_el": np.asarray(res[2], dtype=np.float32).ravel(),
                "vm_nn": np.asarray(res[3], dtype=np.float32).ravel(),
                "se": float(np.max(res[1])),
            })
    return res


def acao_aleatoria_legal(env, rng, tentativas=30):
    """sorteia um elemento solido, fora de apoio/carga, cuja remocao nao desconecta."""
    void = np.asarray(env.VoidCheck, dtype=float).ravel()
    proibidos = set(int(b) for b in np.asarray(env.BC).ravel())
    cand = [i for i in np.where(void == 1)[0] if int(i) not in proibidos]
    if not cand:
        return None
    for i in rng.permutation(cand)[:tentativas]:
        teste = void.copy()
        teste[i] = 0
        if isolate_largest_group_original(np.reshape(teste, (env.EX, env.EY)))[1]:
            return int(i)
    return None


# ---------------------------------------------------------------------------
# casos e episodios
# ---------------------------------------------------------------------------

def sorteia_caso(rng):
    """apoios em uma ou duas arestas e carga vertical ou horizontal sorteada.

    apoios ficam na borda (BC_Nodes de brown so trata elementos de aresta);
    a carga pode cair no interior, como no console do benchmark.
    """
    arestas = [_BORDA_INF, _BORDA_SUP, _BORDA_ESQ, _BORDA_DIR]
    n_ar = int(rng.integers(1, 3))
    escolhidas = rng.choice(4, size=n_ar, replace=False)
    bcs = []
    for a in escolhidas:
        bcs += [int(b) for b in rng.choice(arestas[a], size=int(rng.integers(1, 4)), replace=False)]
    bcs = sorted(set(bcs))

    n_carga = int(rng.integers(1, 4))
    if rng.random() < 0.7:
        base = list(arestas[int(rng.integers(0, 4))])
    else:
        base = [EX * y + x + 1 for y in range(2, EY - 2) for x in range(2, EX - 2)]
    ini = int(rng.integers(0, len(base) - n_carga + 1))
    cargas = [c for c in base[ini:ini + n_carga] if c not in bcs]
    if not cargas:
        cargas = [int(rng.choice([c for c in base if c not in bcs]))]

    conf = {"bcs": bcs, "downs": [], "rights": [],
            "vol": float(rng.uniform(*VOL_ALEATORIO))}
    conf["downs" if rng.random() < 0.6 else "rights"] = cargas
    return conf


def roda_episodio(tarefa):
    """um episodio completo do dqn de brown com mef real, gravando a malha 24x24."""
    ep_id, caso_id, conf, eps, semente = tarefa
    random.seed(semente)
    np.random.seed(semente)
    rng = np.random.default_rng(semente)

    full_conf = {
        "bcs":   [str(b) for b in conf["bcs"]],
        "downs": [str(d) for d in conf.get("downs", [])],
        "rights": [str(r) for r in conf.get("rights", [])],
        "lefts": [], "ups": [],
        "volfraction": conf["vol"],
    }
    test_args = ["main.py", "--From_App", "True", "--Load_Checkpoints", "True",
                 "--Progressive_Refinement", "True", "--Time_Trial", "True"]

    GRAVADOR["ativo"] = True
    GRAVADOR["amostras"] = []
    GRAVADOR["vistos"] = set()

    escolha_orig = Agent.choose_action

    def escolha_eps(self, observation, load_checkpoint, Testing):
        if eps > 0 and rng.random() < eps:
            a = acao_aleatoria_legal(self.env, rng)
            if a is not None:
                return a
        return escolha_orig(self, observation, load_checkpoint, Testing)

    t0 = time.perf_counter()
    status = "ok"
    try:
        with patch.object(sys, "argv", test_args):
            opts = parse_opts()
        envs = EnviromentsRL(opts)
        with patch("Top_Opt_RL.DQN.main.FEASolve", side_effect=fea_gravador), \
             patch("Top_Opt_RL.DQN.TopOpt_Env_Functions.FEASolve", side_effect=fea_gravador), \
             patch.object(Agent, "choose_action", escolha_eps):
            TopOpt_Designing(full_conf, opts, envs, [])
    except SystemExit:
        status = "combinacao ilegal de apoio e carga (sys.exit de brown)"
    except Exception as e:   # caso sorteado que o codigo de brown nao trata
        status = f"erro: {type(e).__name__}: {e}"
    finally:
        GRAVADOR["ativo"] = False
        tf.keras.backend.clear_session()

    return {"ep_id": ep_id, "caso": caso_id, "eps": eps, "status": status,
            "tempo": time.perf_counter() - t0, "amostras": GRAVADOR["amostras"]}


def monta_tarefas(rapido, n_teste=N_EPIS_CASO_TESTE, n_alea=N_CASOS_ALEATORIOS,
                  n_ep_alea=N_EPIS_CASO_ALEATORIO):
    rng = np.random.default_rng(SEED)
    tarefas = []
    n_teste = 2 if rapido else n_teste
    for c, (nome, conf) in enumerate(sorted(TEST_CASES.items())):
        if rapido and c > 0:
            break
        epss = [0.0] + list(rng.uniform(EPS_MIN, EPS_MAX, size=n_teste - 1))
        for eps in epss:
            tarefas.append((len(tarefas), c, conf, float(eps), int(rng.integers(10**9))))
    n_alea = 2 if rapido else n_alea
    n_ep_alea = 1 if rapido else n_ep_alea
    for k in range(n_alea):
        conf = sorteia_caso(rng)
        epss = [0.0] + list(rng.uniform(EPS_MIN, EPS_MAX, size=n_ep_alea - 1))
        for eps in epss:
            tarefas.append((len(tarefas), 3 + k, conf, float(eps), int(rng.integers(10**9))))
    return tarefas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="teste de fumaca: 2 episodios do cantilever e 2 casos sorteados")
    ap.add_argument("--workers", type=int, default=1,
                    help="processos em paralelo (cada um carrega o tensorflow)")
    ap.add_argument("--epis_teste", type=int, default=N_EPIS_CASO_TESTE,
                    help="episodios por caso do benchmark (1 de referencia + perturbados)")
    ap.add_argument("--casos", type=int, default=N_CASOS_ALEATORIOS,
                    help="combinacoes de apoio/carga sorteadas")
    ap.add_argument("--epis_aleatorio", type=int, default=N_EPIS_CASO_ALEATORIO,
                    help="episodios por caso sorteado (eps = 0 e eps sorteados)")
    args = ap.parse_args()

    _HERE = os.path.dirname(os.path.abspath(__file__))
    # o --rapido grava no mesmo arquivo (teste rapido de 04b e 04c); a coleta
    # completa sobrescreve
    save_path = os.path.join(_HERE, "resultados", "dados", "brown_tensao_24x24.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tarefas = monta_tarefas(args.rapido, args.epis_teste, args.casos, args.epis_aleatorio)
    print(f"coletando dados do substituto de campo — mef de brown 24x24 | "
          f"episodios: {len(tarefas)} | workers: {args.workers}")
    print(f"casos do benchmark: {args.epis_teste if not args.rapido else 2} episodios cada "
          f"(eps = 0 e eps em [{EPS_MIN}, {EPS_MAX}]) | casos sorteados: "
          f"{args.casos if not args.rapido else 2} x {args.epis_aleatorio} episodios")

    resultados = []
    t0 = time.time()
    if args.workers <= 1:
        for k, tarefa in enumerate(tarefas):
            r = roda_episodio(tarefa)
            resultados.append(r)
            print(f"  episodio {k+1}/{len(tarefas)} | caso {r['caso']} | eps {r['eps']:.2f} | "
                  f"{len(r['amostras'])} estados | {r['tempo']:.0f}s | {r['status']}")
    else:
        # um thread de blas por processo para nao disputar nucleos entre os workers
        for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            os.environ[v] = "1"
        # spawn: o tensorflow ja foi importado no processo pai e nao convive com fork
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(roda_episodio, t) for t in tarefas]
            for k, f in enumerate(as_completed(futs)):
                r = f.result()
                resultados.append(r)
                print(f"  episodio {k+1}/{len(tarefas)} | caso {r['caso']} | eps {r['eps']:.2f} | "
                      f"{len(r['amostras'])} estados | {r['tempo']:.0f}s | "
                      f"{time.time() - t0:.0f}s total | {r['status']}")

    # junta tudo sem estados repetidos. os episodios de referencia vao por
    # ultimo: um estado que dividem com um episodio perturbado (como o bloco
    # cheio de cada caso) fica com o perturbado e entra no treino
    resultados.sort(key=lambda r: (r["caso"] < 3 and r["eps"] == 0.0, r["ep_id"]))
    solido_l, idx_l, vm_el_l, vm_nn_l, se_l = [], [], [], [], []
    ep_l, caso_l, eps_l, ref_l = [], [], [], []
    mapas_l, mapas_idx = [], {}
    vistos = set()
    for r in resultados:
        for a in r["amostras"]:
            chave_m = a["mapas"].tobytes()
            chave = a["solido"].tobytes() + chave_m
            if chave in vistos:
                continue
            vistos.add(chave)
            if chave_m not in mapas_idx:
                mapas_idx[chave_m] = len(mapas_l)
                mapas_l.append(a["mapas"])
            solido_l.append(a["solido"])
            idx_l.append(mapas_idx[chave_m])
            vm_el_l.append(a["vm_el"])
            vm_nn_l.append(a["vm_nn"])
            se_l.append(a["se"])
            ep_l.append(r["ep_id"])
            caso_l.append(r["caso"])
            eps_l.append(r["eps"])
            ref_l.append(r["caso"] < 3 and r["eps"] == 0.0)

    np.savez_compressed(save_path,
                        solido=np.array(solido_l, dtype=np.uint8),
                        mapa_idx=np.array(idx_l, dtype=np.int32),
                        mapas=np.array(mapas_l, dtype=np.float32),
                        vm_el=np.array(vm_el_l, dtype=np.float32),
                        vm_nn=np.array(vm_nn_l, dtype=np.float32),
                        se=np.array(se_l, dtype=np.float64),
                        episodio=np.array(ep_l, dtype=np.int32),
                        caso=np.array(caso_l, dtype=np.int32),
                        eps=np.array(eps_l, dtype=np.float32),
                        ref=np.array(ref_l, dtype=bool))

    n_ok = sum(r["status"] == "ok" for r in resultados)
    se = np.array(se_l)
    vf = np.array([s.mean() for s in solido_l])
    print(f"\ncoleta concluida: {len(solido_l)} estados unicos | {n_ok}/{len(resultados)} "
          f"episodios ok | {time.time() - t0:.0f}s")
    print(f"combinacoes de apoio/carga: {len(mapas_l)} | estados de referencia: {sum(ref_l)}")
    if len(se):
        print(f"energia min: {se.min():.3g} | max: {se.max():.3g} | "
              f"fracao de volume: {vf.min():.2f} a {vf.max():.2f}")
    print(f"salvo: {save_path}")
