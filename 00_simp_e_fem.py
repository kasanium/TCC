"""
Solver MEF + SIMP para otimizacao topologica de minima compliancia.

Implementacao propria em Python das rotinas classicas de otimizacao
topologica baseada em densidades (Sigmund, 2001; Andreassen et al., 2011;
Bendsoe e Sigmund, 2004), com montagem esparsa e operacoes vetorizadas.

Validacao: caso cantilever de Kharmanda et al. (2004).

Funcoes publicas:
    lk()                                  matriz de rigidez do elemento Q4
    precompute_filter(nelx, nely, rmin)   filtro de sensibilidade (H, Hs)
    FE(nelx, nely, x, penal, ...)         resolve KU = F, retorna U
    sensitivity_analysis(...)             retorna (c, dc, ce_map, vm_map)
    optimality_criteria_update(...)       atualizacao por criterio de otimalidade

Classes auxiliares para o ambiente de aprendizado por reforco:
    FEMSolver       compliancia com densidades intermediarias (SIMP classico)
    FEMSolverRL     compliancia binaria U^T F (formulacao de Brown et al., 2022)
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.linewidth": 1.5,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def formatar_e_salvar(fig, ax, titulo, caminho_saida, is_image=True):
    ax.set_title(titulo, pad=15)
    if is_image:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(caminho_saida)
    plt.close(fig)
    print(f"Arquivo salvo em: {caminho_saida}")


def lk():
    """Matriz de rigidez do elemento quadrilatero bilinear (Q4).

    Forma analitica para material isotropico (Sigmund, 2001;
    Andreassen et al., 2011).
    """
    E, nu = 1.0, 0.3
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    return E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


def precompute_filter(nelx, nely, rmin):
    """Pre-calcula o operador de filtro de sensibilidade H e a soma das
    linhas Hs (Sigmund, 2001).

    A matriz H e densa; e suficiente para as malhas usadas neste trabalho,
    mas escala mal em malhas muito grandes.
    """
    if rmin <= 0:
        print("aviso: rmin <= 0, filtro inativo na pratica")
    H = np.zeros((nelx * nely, nelx * nely))
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - int(rmin), 0), min(i1 + int(rmin) + 1, nelx)):
                for j2 in range(max(j1 - int(rmin), 0), min(j1 + int(rmin) + 1, nely)):
                    e2 = i2 * nely + j2
                    dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
                    H[e1, e2] = max(0, rmin - dist)
    return H, np.sum(H, 1)


def _build_indices(nelx, nely):
    """Indices de montagem esparsa: (edofMat, iK, jK)."""
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[ely + elx * nely] = [
                2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1,
                2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]
    iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
    return edofMat, iK, jK


def FE(nelx, nely, x, penal, fixed_dofs=None, force_vector=None):
    """Resolve KU = F e retorna U.

    x pode ter shape (nely*nelx,) ou (nely, nelx). fixed_dofs e
    force_vector sao opcionais; None aplica o cantilever padrao (engaste
    na borda esquerda, carga unitaria no canto inferior direito).
    """
    KE = lk()
    ndof = 2 * (nelx + 1) * (nely + 1)
    x_flat = np.asarray(x).flatten(order="F")
    edofMat, iK, jK = _build_indices(nelx, nely)

    # penalizacao SIMP com Emin = 0 (x^p puro)
    E_vec = x_flat ** penal
    sK = ((KE.flatten()[np.newaxis]).T * E_vec).flatten(order="F")
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    if fixed_dofs is None:
        fixed_dofs = np.arange(2 * (nely + 1))
    if force_vector is None:
        force_vector = np.zeros(ndof)
        force_vector[ndof - 1] = -1.0

    freedofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    U = np.zeros(ndof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U[freedofs] = spsolve(K[freedofs, :][:, freedofs], force_vector[freedofs])
    return U


def sensitivity_analysis(nelx, nely, x, penal, U):
    """Retorna (c, dc, ce_map, vm_map).

    c        compliancia total (escalar)
    dc       sensibilidade por elemento, shape (nely*nelx,)
    ce_map   energia de deformacao por elemento, shape (nely, nelx)
    vm_map   von Mises aproximado por elemento, shape (nely, nelx)
    """
    KE = lk()
    x_flat = np.asarray(x).flatten(order="F")
    edofMat, _, _ = _build_indices(nelx, nely)
    Ue = U[edofMat]                                     # (n_elem, 8)

    # energia de deformacao por elemento (Andreassen et al., 2011)
    ce_flat = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
    E_vec = x_flat ** penal
    c = float(np.dot(E_vec, ce_flat))

    # sensibilidade dc/dx_e = -p * x_e^(p-1) * ce (Sigmund, 2001)
    dc = -penal * (x_flat ** (penal - 1)) * ce_flat

    # energia efetiva (com penalizacao) por elemento
    ce_map = (ce_flat * E_vec).reshape((nely, nelx), order="F")

    # von Mises aproximado no centro do elemento Q4
    nu = 0.3
    C_mat = (1.0 / (1 - nu**2)) * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1 - nu) / 2]])
    B_mat = 0.5 * np.array([
        [-1,  0,  1,  0,  1,  0, -1,  0],
        [ 0, -1,  0, -1,  0,  1,  0,  1],
        [-1, -1, -1,  1,  1,  1,  1, -1]])
    stress = np.einsum("ij,jk,ek->ei", C_mat, B_mat, Ue) * E_vec[:, np.newaxis]
    s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 2]
    vm_flat = np.sqrt(np.maximum(0.0, s11**2 + s22**2 - s11*s22 + 3*s12**2))
    vm_map = vm_flat.reshape((nely, nelx), order="F")

    return c, dc, ce_map, vm_map


def optimality_criteria_update(nelx, nely, x, volfrac, dc, H, Hs):
    """Atualizacao das densidades por criterio de otimalidade (Sigmund, 2001).

    O multiplicador de Lagrange lambda e obtido por bisseccao para satisfazer
    a restricao de volume.
    """
    dc = (H @ (x * dc)) / (Hs * x)
    l1, l2 = 0.0, 100000.0
    move = 0.2
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001,
               np.maximum(x - move,
               np.minimum(1.0,
               np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        if xnew.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


class FEMSolver:
    """Wrapper das funcoes do modulo para uso pelo ambiente de otimizacao.

    Condicoes de contorno fixas: engaste na borda esquerda, carga vertical
    no meio da aresta direita (configuracao classica de cantilever).
    Retorna ce_map com a penalizacao x^p embutida, coerente com o SIMP.
    """

    def __init__(self, nelx, nely, vol_frac=0.5, penal=3.0, rmin=1.5):
        self.nelx = nelx
        self.nely = nely
        self.penal = penal
        self.rmin = rmin
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        self.KE = lk()
        self.edofMat, self.iK, self.jK = _build_indices(nelx, nely)
        self.H, _Hs = precompute_filter(nelx, nely, rmin)

        self.fixeddofs = np.arange(0, 2 * (nely + 1))
        self.freedofs = np.setdiff1d(np.arange(self.ndof), self.fixeddofs)

        self.F = np.zeros(self.ndof)
        node_mid_right = nelx * (nely + 1) + nely // 2
        self.F[2 * node_mid_right + 1] = -1.0

    def solve(self, x):
        U = FE(self.nelx, self.nely, x, self.penal,
               fixed_dofs=self.fixeddofs, force_vector=self.F)
        _, _, ce_map, vm_map = sensitivity_analysis(
            self.nelx, self.nely, x, self.penal, U)
        return U, ce_map, vm_map


class FEMSolverRL:
    """Solver MEF para o ambiente de aprendizado por reforco.

    Diferenca em relacao a FEMSolver: aqui a compliancia e calculada como
    U^T F, coerente com a formulacao de Brown et al. (2022), que usa acoes
    binarias (E_e em {E0, Emin}) sem densidades intermediarias. A penalizacao
    x^p nao tem papel mecanico nesse regime e distorceria a escala da
    recompensa.

    API:
        solve(x) -> (U, compliance, vm_map)
            compliance : float = U^T F
            vm_map     : ndarray (nely, nelx), von Mises para o canal de
                         observacao do agente

    Condicoes de contorno: engaste na borda esquerda, carga vertical no meio
    da aresta direita (configuracao classica de cantilever).
    """

    def __init__(self, nelx, nely, vol_frac=0.5, penal=3.0, rmin=1.5):
        self.nelx = nelx
        self.nely = nely
        self.penal = penal
        self.rmin = rmin
        self.Emin = 1e-9
        self.E0 = 1.0
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        self.KE = self._lk()
        self.iK, self.jK, self.edofMat = self._build_indices()
        self.freedofs, self.F = self._build_bcs()

    def _lk(self):
        nu = 0.3
        k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                      -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
        return 1.0 / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    def _build_indices(self):
        edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                edofMat[el] = [2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1,
                               2*n2 + 2, 2*n2 + 3, 2*n1 + 2, 2*n1 + 3]
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()
        return iK, jK, edofMat

    def _build_bcs(self):
        # engaste na borda esquerda; carga no meio da aresta direita
        fixed_nodes = np.arange(0, self.nely + 1)
        fixeddofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        freedofs = np.setdiff1d(np.arange(self.ndof), fixeddofs)

        F = np.zeros(self.ndof)
        load_node = (self.nely + 1) * self.nelx + self.nely // 2
        F[2 * load_node + 1] = -1.0
        return freedofs, F

    def solve(self, x):
        """Resolve KU = F e retorna (U, compliance, vm_map).

        compliance = U^T F = sum_e E_e * (u_e^T KE u_e).

        Nao aplica filtro de densidade: o agente trabalha com geometria
        binaria, e filtrar borraria essa geometria.
        """
        x_flat = x.flatten(order="F")
        E_vec = self.Emin + (x_flat ** self.penal) * (self.E0 - self.Emin)
        sK = ((self.KE.flatten()[np.newaxis]).T * E_vec).flatten(order="F")
        K = coo_matrix((sK, (self.iK, self.jK)),
                       shape=(self.ndof, self.ndof)).tocsc()
        K_red = K[self.freedofs, :][:, self.freedofs]

        U = np.zeros(self.ndof)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                U[self.freedofs] = spsolve(K_red, self.F[self.freedofs])
            except Exception:
                return U, float("inf"), np.zeros((self.nely, self.nelx))

        Ue = U[self.edofMat]
        ce_flat = np.einsum("ij,jk,ik->i", Ue, self.KE, Ue)
        ce = np.maximum(ce_flat * E_vec, 0.0)
        compliance = float(np.sum(ce))            # = U^T F

        # von Mises no centro do elemento Q4
        nu = 0.3
        C_mat = (1.0 / (1 - nu**2)) * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1 - nu) / 2]])
        B_mat = 0.5 * np.array([
            [-1,  0,  1,  0,  1,  0, -1,  0],
            [ 0, -1,  0, -1,  0,  1,  0,  1],
            [-1, -1, -1,  1,  1,  1,  1, -1]])
        stress = np.einsum("ij,jk,ek->ei", C_mat, B_mat, Ue) * E_vec[:, np.newaxis]
        s11, s22, s12 = stress[:, 0], stress[:, 1], stress[:, 2]
        vm_flat = np.sqrt(np.maximum(0.0, s11**2 + s22**2 - s11*s22 + 3*s12**2))
        vm_map = vm_flat.reshape((self.nely, self.nelx), order="F")

        return U, compliance, vm_map


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(_HERE, "resultados"), exist_ok=True)

    # caso00: validacao contra Kharmanda et al. (2004); caso02: malha do projeto
    casos = [
        {
            "nome": "caso00_original",
            "nelx": 32, "nely": 20, "volfrac": 0.4, "penal": 3.0, "rmin": 1.2,
            "carga": "canto_inferior_direito",
        },
        {
            "nome": "caso02_novo",
            "nelx": 60, "nely": 30, "volfrac": 0.5, "penal": 3.0, "rmin": 3.0,
            "carga": "meio_direita",
        },
    ]

    for caso in casos:
        nelx = caso["nelx"]
        nely = caso["nely"]
        volfrac = caso["volfrac"]
        penal = caso["penal"]
        rmin = caso["rmin"]

        _OUT = os.path.join(_HERE, "resultados", f'00_validacao_{caso["nome"]}.png')
        print(f"\n{'='*50}")
        print(f"Iniciando {caso['nome']} ({nelx}x{nely})")
        print(f"{'='*50}")

        ndof = 2 * (nelx + 1) * (nely + 1)
        F = np.zeros(ndof)
        if caso["carga"] == "canto_inferior_direito":
            F[ndof - 1] = -1.0
        elif caso["carga"] == "meio_direita":
            node_mid_right = nelx * (nely + 1) + nely // 2
            F[2 * node_mid_right + 1] = -1.0

        x = volfrac * np.ones(nely * nelx, dtype=float)
        loop = 0
        change = 1.0
        H, Hs = precompute_filter(nelx, nely, rmin)

        # iteracao 0 (bloco cheio homogeneo)
        U_zero = FE(nelx, nely, x, penal, force_vector=F)
        c_zero, _, _, _ = sensitivity_analysis(nelx, nely, x, penal, U_zero)
        print(f"iter:   0 | obj: {c_zero:6.3f} | vol: {x.mean():.3f} | mudanca: ------")

        while change > 0.01 and loop < 100:
            loop += 1
            xold = x.copy()
            U = FE(nelx, nely, x, penal, force_vector=F)
            c, dc, _, _ = sensitivity_analysis(nelx, nely, x, penal, U)
            x = optimality_criteria_update(nelx, nely, x, volfrac, dc, H, Hs)
            change = np.linalg.norm(x - xold, np.inf)
            if loop % 5 == 0 or change <= 0.01:
                print(f"iter: {loop:3d} | obj: {c:6.3f} | vol: {x.mean():.3f} | mudanca: {change:.4f}")

        print(f"\nCompliance final ({caso['nome']}): {c:.4f}")

        _, _, ce_map, vm_map = sensitivity_analysis(nelx, nely, x, penal, U)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        axes[0].imshow(x.reshape((nelx, nely)).T, cmap="gray_r",
                       interpolation="none", aspect="equal")
        axes[0].set_title(f"(a) Topologia | C = {c:.4f}", fontweight="bold")

        vm_norm = vm_map / (vm_map.max() + 1e-12)
        im2 = axes[1].imshow(vm_norm, cmap="turbo", interpolation="none",
                             aspect="equal", vmin=0, vmax=1)
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb2.set_label("Von Mises normalizado", fontsize=9)
        cb2.ax.tick_params(labelsize=8)
        axes[1].set_title("(b) Von Mises (aprox.)", fontweight="bold")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("black")
                sp.set_linewidth(1.5)

        fig.suptitle(
            rf"SIMP — Cantilever {nelx}×{nely} | $E=1,\;\nu=0.3,\;p={penal},\;r_{{min}}={rmin}$",
            fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(_OUT, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Arquivo salvo em: {_OUT}")