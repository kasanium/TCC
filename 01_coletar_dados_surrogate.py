"""
Geracao do banco de dados para treino dos modelos substitutos.

Metodologia: amostragem de iteracoes intermediarias do SIMP em vez de
topologias aleatorias (Banga et al., 2018). Amostragem aleatoria pura gera
arranjos ineficientes e desconexos que degradam a regressao (Zhang et al.,
2024); capturar iteracoes do SIMP preserva caminhos de carga coerentes.

Caso de teste: cantilever 24x24, compativel com o ambiente de RL usado no
treinamento (Brown et al., 2022).

Orcamento: 300 execucoes x 20 capturas = 6000 amostras. O schedule de
captura segue o "Dataset 1" de Banga et al. (2018).

Saida: resultados/dados/topo_data_24x24.npz
    campos: x, y, fx, fy, load_node, run_id, rmin, nelx, nely
"""

import os
import sys
import time
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
from scipy.ndimage import convolve

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

# o solver vive em 00_simp_e_fem.py; o nome comeca com digito e nao pode ser
# importado diretamente, entao carregamos o modulo pelo caminho do arquivo
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "00_simp_e_fem.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FEMSolverRL = _mod.FEMSolverRL


# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------

nelx, nely = 24, 24      # malha quadrada compativel com o ambiente de RL
n_runs = 300             # 300 execucoes x 20 capturas = 6000 amostras
samp_per_run = 20
max_iter = 100           # o SIMP converge em ~70-100 iteracoes (Banga et al., 2018)
seed = 42

# p fixo em 3.0 (valor canonico do SIMP; Andreassen et al., 2011). Variar p sem
# passa-lo como feature criaria ruido nos rotulos, pois a compliancia depende de p.
penal = 3.0

# rmin sorteado por execucao para diversificar as texturas topologicas do
# dataset sem alterar a compliancia de cada amostra capturada.
RMIN_OPTIONS = [1.2, 1.5, 2.0, 2.5]

# cutoff de compliancia: captura todo o espectro operacional do SIMP. As
# iteracoes iniciais (malha homogenea) tem C ~ 320; cortar baixo descartaria os
# dados de alta compliancia necessarios para variancia. Como capturamos
# iteracoes intermediarias (e nao topologias finais), o teto e alto de proposito.
c_cutoff = 320.0

rng = np.random.default_rng(seed)


def build_filter_kernel(rmin):
    """Kernel 2D do filtro de densidade por convolucao (normalizado)."""
    r = int(np.ceil(rmin))
    xx, yy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    kernel = np.maximum(0, rmin - np.sqrt(xx**2 + yy**2))
    return kernel / np.sum(kernel)


def banga_iter_schedule(n_samples, max_iter, rng):
    """Schedule de captura em duas etapas (Banga et al., 2018, Dataset 1).

    Etapa 1: sorteia o limite superior m ~ Uniforme(1, max_iter).
    Etapa 2: sorteia a iteracao alvo n ~ Uniforme(1, m).

    A composicao pesa as iteracoes iniciais, onde a compliancia ainda e alta e
    a variancia e grande, o que o regressor precisa para nao colapsar por
    homogeneidade.
    """
    iters = []
    attempts = 0
    while len(iters) < n_samples * 3 and attempts < n_samples * 50:
        m = int(rng.integers(1, max_iter + 1))
        n = int(rng.integers(1, m + 1))
        iters.append(n)
        attempts += 1
    unique = sorted(set(iters))
    # garante que o SIMP evolua o suficiente, evitando schedules de max baixo
    if not unique or max(unique) < max_iter // 4:
        unique.append(max_iter // 2)
    return unique[:n_samples]


def run_simp_sampled(nelx, nely, vol_frac, rmin, load_iy, fx, fy, sample_iters):
    """Executa o SIMP e captura snapshots nas iteracoes sorteadas.

    Usa FEMSolverRL para resolver KU = F e obter compliancia U^T F. O laco de
    criterio de otimalidade (Sigmund, 2001) e conduzido aqui, com filtro de
    densidade aplicado por convolucao.
    """
    solver = FEMSolverRL(nelx, nely, vol_frac=vol_frac, penal=penal, rmin=rmin)

    # condicoes de contorno do cantilever com carga variavel (sobrescreve o default)
    fixed_nodes = np.arange(0, nely + 1)
    fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    solver.fixeddofs = np.unique(fixed_dofs)
    solver.freedofs = np.setdiff1d(np.arange(solver.ndof), solver.fixeddofs)
    load_node = (nely + 1) * nelx + load_iy
    solver.F = np.zeros(solver.ndof)
    solver.F[2 * load_node] = fx
    solver.F[2 * load_node + 1] = fy

    x = vol_frac * np.ones((nely, nelx))    # inicializacao uniforme (Andreassen et al., 2011)
    s_set = set(sample_iters)
    H = build_filter_kernel(rmin)

    samples = []
    for loop in range(1, max(sample_iters) + 1):
        U, c_raw, _ = solver.solve(x)

        # sensibilidade filtrada (Sigmund, 2001)
        xp = np.clip(convolve(x, H, mode="reflect"), 0.001, 1.0)
        ue = U[solver.edofMat]
        ce = np.einsum("ij,jk,ik->i", ue, solver.KE, ue).reshape((nely, nelx), order="F")
        dc = -penal * (xp ** (penal - 1)) * ce
        dc = convolve(dc, H, mode="reflect")

        # atualizacao por criterio de otimalidade (Sigmund, 2001)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) > 1e-4:
            lm = 0.5 * (l2 + l1)
            B = np.sqrt(np.maximum(0.0, -dc / lm))
            xn = np.maximum(0.001, np.maximum(x - 0.2,
                 np.minimum(1.0, np.minimum(x + 0.2, x * B))))
            if np.mean(xn) - vol_frac > 0:
                l1 = lm
            else:
                l2 = lm
        x = xn.copy()

        if loop in s_set:
            # captura o estado fisico com o filtro aplicado
            xp_s = np.clip(convolve(x, H, mode="reflect"), 0.001, 1.0)
            _, c_s, _ = solver.solve(xp_s)
            if np.isfinite(c_s) and 0.0 < c_s < c_cutoff:
                samples.append((xp_s.astype(np.float32).copy(), float(c_s)))

    return samples


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(_HERE, "resultados", "dados", "topo_data_24x24.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"coletando dados surrogate — cantilever {nelx}x{nely} | execucoes: {n_runs}")
    print(f"schedule: duas etapas (Banga 2018) | max_iter={max_iter} | c_cutoff={c_cutoff}")
    print(f"vol_frac: [0.3, 0.8] | rmin: {RMIN_OPTIONS} | penal={penal}")

    x_list, y_list, fx_list, fy_list = [], [], [], []
    node_list, run_id_list, rmin_list = [], [], []
    t0 = time.time()

    # vol_frac sorteado em [0.3, 0.8] para diversificar a topologia e a faixa de C
    vol_fracs = rng.uniform(0.3, 0.8, size=n_runs)
    rmins = rng.choice(RMIN_OPTIONS, size=n_runs)

    # carga: posicao Y na aresta direita e direcao (angulo) sorteadas, magnitude
    # unitaria. Diversificar carga gera diversidade de caminhos de carga.
    load_iys = rng.integers(0, nely + 1, size=n_runs)
    thetas = rng.uniform(0.0, 2 * np.pi, size=n_runs)
    fxs = np.cos(thetas).astype(np.float32)
    fys = np.sin(thetas).astype(np.float32)

    for i in range(n_runs):
        sched = banga_iter_schedule(samp_per_run, max_iter, rng)
        load_iy = int(load_iys[i])
        fx = float(fxs[i])
        fy = float(fys[i])
        vf = float(vol_fracs[i])
        rm = float(rmins[i])

        samples = run_simp_sampled(nelx, nely, vf, rm, load_iy, fx, fy, sched)

        load_node = (nely + 1) * nelx + load_iy
        for xs, cs in samples:
            x_list.append(xs)
            y_list.append(cs)
            fx_list.append(fx)
            fy_list.append(fy)
            node_list.append(load_node)
            run_id_list.append(i)    # grupo para evitar vazamento entre treino/teste
            rmin_list.append(rm)

        if (i + 1) % 20 == 0:
            print(f"  execucao {i+1}/{n_runs} | amostras validas: {len(y_list)}")

    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)
    FX = np.array(fx_list, dtype=np.float32)
    FY = np.array(fy_list, dtype=np.float32)
    NODE = np.array(node_list, dtype=np.int32)
    RUN_ID = np.array(run_id_list, dtype=np.int32)
    RMIN = np.array(rmin_list, dtype=np.float32)

    np.savez_compressed(save_path, x=X, y=Y, fx=FX, fy=FY,
                        load_node=NODE, run_id=RUN_ID, rmin=RMIN,
                        nelx=nelx, nely=nely)

    print(f"\ncoleta concluida: {len(X)} amostras | {time.time() - t0:.1f}s")
    print(f"C min: {Y.min():.2f} | C max: {Y.max():.2f} | std: {Y.std():.2f}")