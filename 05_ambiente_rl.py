"""
ambiente de rl com hard action masking de conectividade (huang e ontanon
2022; hong et al. 2025). a mascara e calculada por pontos de articulacao
(tarjan, o(v+e)), cacheada por estado, com terminacao limpa por deadlock
geometrico (sem acao valida antes do vf alvo -> done, sem nan). observacao
de 5 canais: densidade, tensao normalizada (p99), posto normalizado da
tensao (o argmin dele entre as acoes validas e a politica gulosa eso),
engastes e carga. recompensa telescopica r_t = k*log(c_{t-1}/c_t), cuja
soma no episodio e o proprio objetivo (potential-based shaping, ng, harada
e russell 1999); treino com gamma=1.0. reward_mode="brown" reproduz a eq. 4
de brown et al. (2022). simetria="horizontal" (opcional) remove cada
elemento junto do espelho e so libera pares que preservam um unico
componente; requer bcs simetricas (assert no __init__). compliancia u^t f
via femsolverrl (brown et al. 2022). thin_threshold (espessura minima)
existe mas fica desligado por padrao: medido inerte em 0.00-0.45 e
travando episodios em ~0.60 (ver testes em 07_testes_ambiente.py).

rodar_guloso() e o baseline eso (xie e steven 1993) no mesmo espaco de
acoes do agente: a cada passo remove o elemento valido de menor tensao,
com filtro de sensibilidade opcional (sigmund 2001). o raio do filtro nao
transfere entre malhas: em 30x30 filtrar melhora (22.78 -> 20.82 na geometria
anterior a 2026-10; 20.42 com os furos simetricos), em
20x10 piora (71.66 -> 82.45); sob simetria o efeito inverte de novo.

penalidade de overhang opcional (dfam_peso > 0): a cada passo a recompensa
recebe -dfam_peso*(viol_t - viol_{t-1}), onde viol e a fracao de elementos
solidos em violacao da regra de 45 graus para a base de impressao
dfam_base (mesma metrica de 12_analises_automotivo.py). a soma no episodio
telescopa para -dfam_peso*(viol_T - viol_0), entao o retorno vira
k*log(c_0/c_T) - dfam_peso*viol_T (viol_0 e a do dominio inicial), um
trade-off explicito rigidez x fabricabilidade. isto implementa a
penalidade de fabricacao prevista nos objetivos do projeto: a regra de
45 graus e a de ibhadode et al. (2023) e dangal e jung (2023), que
tambem medem o custo em compliancia de restringir overhang na ot
classica; incorporar a restricao como termo de penalidade no objetivo e
o metodo de penalidade usual; e penalizar o incremento a cada passo (em
vez de um termo esparso so no fim) segue a recomendacao de recompensas
densas para problemas fisicos de mozaffar, ebrahimi e cao (2020). a
soma por episodio equivale ao termo terminal; o peso e calibrado
empiricamente por varredura. desligada por padrao (dfam_peso=0), custo
zero quando desligada.
"""

import os
import time
import numpy as np
import scipy.ndimage
import gymnasium as gym
from gymnasium import spaces
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FEMSolverRL = _mod.FEMSolverRL

# espessura do olhal (em elementos) ao redor de cada furo; identica a do
# 10_simp_automotivo.py
RING_AUTO = 2

# base de impressao -> (sinal de avanco, eixo); referencial: linha 0 = topo
_DFAM_DIRS = {"baixo": (+1, "linha"), "cima": (-1, "linha"),
              "esquerda": (+1, "coluna"), "direita": (-1, "coluna")}


class TopoOptMaskEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, nelx=20, nely=10, vol_frac=0.5, penal=3.0, rmin=1.5,
                 problem_type="cantilever",
                 thin_radius=1.5, thin_threshold=0.0,
                 reward_scale=10.0, deadlock_penalty=10.0,
                 reward_mode="log_delta", simetria=None,
                 dfam_peso=0.0, dfam_base="baixo"):
        super().__init__()

        # simetria horizontal opcional (v22; ver docstring do modulo).
        # None desativa (comportamento v21).
        assert simetria in (None, "horizontal"), simetria
        self.simetria = simetria

        # recompensa (ver item 2 da docstring do modulo)
        #   "log_delta" : r_t = reward_scale * log(c_{t-1}/c_t)   [v21, padrao]
        #   "brown"     : r_t = (c_s/c_t)^2 + (alpha_t/N^2)^2     [v19/v20]
        self.reward_scale = float(reward_scale)
        self.deadlock_penalty = float(deadlock_penalty)
        self.reward_mode = reward_mode

        # penalidade de overhang opcional (ver docstring do modulo)
        assert dfam_base in _DFAM_DIRS, dfam_base
        self.dfam_peso = float(dfam_peso)
        self.dfam_base = dfam_base
        self._viol_prev = 0.0

        self.nelx = nelx
        self.nely = nely
        self.target_vol = vol_frac
        self.penal = penal
        self.rmin = rmin
        self.problem_type = problem_type
        self.n_elements = nelx * nely

        # restricao de espessura minima (ver docstring do modulo).
        # thin_threshold=0 desativa a restricao por completo.
        self.thin_radius = thin_radius
        self.thin_threshold = thin_threshold
        self._thin_kernel, self._thin_w1 = self._build_thin_kernel(thin_radius)

        # compliancia = U^T F (Brown et al., 2022)
        self.solver = FEMSolverRL(nelx, nely, vol_frac, penal, rmin)

        # mascaras passivas (no cantilever ambas ficam vazias)
        self.void_mask = np.zeros((self.nely, self.nelx), dtype=bool)   # furo
        self.solid_mask = np.zeros((self.nely, self.nelx), dtype=bool)  # olhal

        if problem_type == "automotive":
            self._setup_automotive_bcs()
        # no cantilever as BCs ja vem do FEMSolverRL

        # alvo de volume e numero de remocoes. Os furos ja nascem vazios, entao a
        # contagem de remocoes parte dos solidos iniciais (n_elements - n_void).
        self.target_n_solid = int(self.n_elements * vol_frac)
        n_void = int(self.void_mask.sum())
        n_solid_start = self.n_elements - n_void
        self.n_to_remove = n_solid_start - self.target_n_solid

        # acao: indice do elemento a remover
        self.action_space = spaces.Discrete(self.n_elements)

        # observacao: 5 canais (densidade, tensao normalizada, posto da
        # tensao, fixos, carga) — ver item 1 da docstring do modulo
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5, nely, nelx), dtype=np.float32)

        # conectividade-4 (vizinhos face a face, sem diagonal)
        self._struct = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])

        self._fixed_channel, self._load_channel = self._build_obs_channels()
        self.protected_mask = self._build_protected_mask()
        self._protected_flat = self.protected_mask.flatten()

        if self.simetria == "horizontal":
            # espelho: mesma coluna, linha refletida em torno do centro
            # vertical. Indice achatado, pre-computado uma vez.
            ely_idx, elx_idx = np.divmod(np.arange(self.n_elements), self.nelx)
            ely_mirror = self.nely - 1 - ely_idx
            self._mirror_flat = ely_mirror * self.nelx + elx_idx
            # o esquema so faz sentido se as BCs (protegidos) forem
            # simetricas; se nao forem, a mascara ficaria incoerente.
            assert np.array_equal(self._protected_flat,
                                  self._protected_flat[self._mirror_flat]), (
                "simetria='horizontal' pedida mas a mascara de protegidos "
                "nao e simetrica para este problem_type/malha")

        # cache da mascara: valida enquanto _mask_dirty for False
        self._mask_cache = None
        self._mask_dirty = True

    def _setup_automotive_bcs(self):
        # dois furos circulares (parafusos): geometria de referencia 30x30 com
        # furos em (5; 7,5) e (5; 22,5), raio 2, escalada em ponto flutuante.
        # Deve ser identica a build_bcs() de 10_simp_automotivo.py, senao SIMP
        # e agente resolvem pecas diferentes.
        scale = self.nelx / 30.0
        centers = [(5.0 * scale, 7.5 * scale),
                   (5.0 * scale, 22.5 * scale)]
        r_hole = 2.0 * scale
        r_boss = r_hole + RING_AUTO * scale

        ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # mascaras de elemento: void (furo) e solid (olhal)
        for ely in range(self.nely):
            for elx in range(self.nelx):
                cx, cy = elx + 0.5, ely + 0.5
                for (hx, hy) in centers:
                    d2 = (cx - hx) ** 2 + (cy - hy) ** 2
                    if d2 <= r_hole ** 2:
                        self.void_mask[ely, elx] = True
                    elif d2 <= r_boss ** 2:
                        self.solid_mask[ely, elx] = True
        self.solid_mask &= ~self.void_mask   # o furo tem prioridade sobre o olhal

        # engaste apenas nos nos da interface do furo. O olhal fica com nos livres,
        # deforma e carrega tensao real, de modo que o canal de von Mises informa o
        # envoltorio do furo. Fixar todo o disco mataria essa tensao. Tambem evita
        # nos livres flutuando dentro do furo, que tornariam K singular.
        fixed_nodes = set()
        for ely, elx in np.argwhere(self.void_mask):
            for (cx, cy) in ((elx, ely), (elx + 1, ely),
                             (elx, ely + 1), (elx + 1, ely + 1)):
                fixed_nodes.add((self.nely + 1) * cx + cy)
        fixed_nodes = np.array(sorted(fixed_nodes), dtype=int)
        self.solver.fixeddofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        self.solver.freedofs = np.setdiff1d(np.arange(ndof), self.solver.fixeddofs)

        # carga no meio da aresta direita
        self.solver.F = np.zeros(ndof)
        load_node = (self.nely + 1) * self.nelx + self.nely // 2
        self.solver.F[2 * load_node + 1] = -1.0

        self._centers = centers
        self._r_hole = r_hole
        self._r_boss = r_boss

    def _build_obs_channels(self):
        # canal 1: elementos com no fixo (Dirichlet); canal 2: elemento carregado.
        # Refletem o problem_type para nao dar BCs erradas ao agente automotivo.
        c1 = np.zeros((self.nely, self.nelx), dtype=np.float32)
        c2 = np.zeros((self.nely, self.nelx), dtype=np.float32)
        mid = self.nely // 2

        if self.problem_type == "cantilever":
            c1[:, 0] = 1.0
            for ely_c in [mid - 1, mid]:
                if 0 <= ely_c < self.nely:
                    c2[ely_c, self.nelx - 1] = 1.0

        elif self.problem_type == "automotive":
            # fixos = interface do furo; o olhal nao entra aqui, pois e material
            # que deforma, nao ancoragem
            c1[self.void_mask] = 1.0
            for ely_c in [mid - 1, mid]:
                if 0 <= ely_c < self.nely:
                    c2[ely_c, self.nelx - 1] = 1.0

        return c1, c2

    def _build_protected_mask(self):
        mask = np.zeros((self.nely, self.nelx), dtype=bool)
        if self.problem_type == "cantilever":
            mask[:, 0] = True
            mid = self.nely // 2
            for ely_prot in [mid - 1, mid]:
                if 0 <= ely_prot < self.nely:
                    mask[ely_prot, self.nelx - 1] = True
        elif self.problem_type == "automotive":
            # olhal e furo protegidos, mais o elemento de carga
            mask |= self.solid_mask
            mask |= self.void_mask
            mid = self.nely // 2
            for ely_prot in [mid - 1, mid]:
                if 0 <= ely_prot < self.nely:
                    mask[ely_prot, self.nelx - 1] = True
        return mask

    @staticmethod
    def _build_thin_kernel(radius):
        """Kernel radial de densidade local (mesma forma de
        precompute_filter()/01_fem_simp.py), normalizado para somar 1, e o
        peso do kernel na distancia unitaria (usado para atualizar a
        densidade local de um vizinho ao remover o elemento central)."""
        r = int(np.ceil(radius))
        yy, xx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        kernel = np.maximum(0.0, radius - np.sqrt(xx ** 2 + yy ** 2))
        kernel = kernel / kernel.sum()
        w1 = kernel[r + 1, r]  # peso a distancia 1 (vizinho face a face)
        return kernel, w1

    def _thin_flags(self, solid, candidates):
        """Para cada candidato a remocao (indice achatado com candidates[i]
        True), verifica se removê-lo deixaria algum vizinho face a face
        ainda solido com densidade local abaixo de thin_threshold. Retorna
        um vetor booleano achatado: True = bloquear (deixaria vizinho fino
        demais). Ver docstring do modulo."""
        thin_flat = np.zeros(self.n_elements, dtype=bool)
        if self.thin_threshold <= 0:
            return thin_flat

        local = scipy.ndimage.convolve(solid.astype(float), self._thin_kernel,
                                       mode="constant", cval=0.0)
        w1 = self._thin_w1
        idxs = np.nonzero(candidates)[0]
        for i in idxs:
            ely, elx = i // self.nelx, i % self.nelx
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = ely + dy, elx + dx
                if 0 <= ny < self.nely and 0 <= nx < self.nelx and solid[ny, nx]:
                    if local[ny, nx] - w1 < self.thin_threshold:
                        thin_flat[i] = True
                        break
        return thin_flat

    def _violacao_overhang(self, solid):
        """fracao de elementos solidos sem suporte a 45 graus para a base
        dfam_base (suportado = na base ou com vizinho solido na camada
        anterior, incluindo diagonais). mesma regra de 12_analises."""
        sgn, eixo = _DFAM_DIRS[self.dfam_base]
        S = solid if eixo == "linha" else solid.T
        nrow = S.shape[0]
        ordem = range(nrow - 1, -1, -1) if sgn == +1 else range(nrow)
        n_viol, prev = 0, None
        for r in ordem:
            linha = S[r]
            if prev is None:
                prev = linha
                continue
            sup = prev.copy()
            sup[1:] |= prev[:-1]
            sup[:-1] |= prev[1:]
            n_viol += int(np.count_nonzero(linha & ~sup))
            prev = linha
        return n_viol / max(int(solid.sum()), 1)

    def _is_connected(self, density):
        # corpo conexo: exatamente um componente solido
        solid = (density > 0.5).astype(int)
        _, n = scipy.ndimage.label(solid, structure=self._struct)
        return n == 1

    def _articulation_flags(self, solid):
        """Pontos de articulacao do grafo de conectividade-4 dos elementos
        solidos, por busca em profundidade iterativa (algoritmo de Tarjan).
        Retorna um vetor booleano achatado: True onde remover o elemento
        fragmentaria o corpo. Custo O(V+E); equivalente exato ao teste
        elemento a elemento com scipy.ndimage.label (07_testes_ambiente.py).
        """
        nely, nelx = solid.shape
        ys, xs = np.nonzero(solid)
        n_solid = len(ys)
        art_flat = np.zeros(nely * nelx, dtype=bool)
        if n_solid <= 2:
            # com 1 ou 2 elementos solidos nenhum e ponto de articulacao;
            # o caso de 0 ou 1 remanescentes e tratado pelo chamador
            return art_flat

        idx = -np.ones((nely, nelx), dtype=int)
        idx[ys, xs] = np.arange(n_solid)

        adj = [[] for _ in range(n_solid)]
        for y, x in zip(ys, xs):
            i = idx[y, x]
            if y + 1 < nely and solid[y + 1, x]:
                j = idx[y + 1, x]
                adj[i].append(j)
                adj[j].append(i)
            if x + 1 < nelx and solid[y, x + 1]:
                j = idx[y, x + 1]
                adj[i].append(j)
                adj[j].append(i)

        disc = [-1] * n_solid
        low = [0] * n_solid
        parent = [-1] * n_solid
        is_art = [False] * n_solid
        timer = 0

        for root in range(n_solid):
            if disc[root] != -1:
                continue
            disc[root] = low[root] = timer
            timer += 1
            stack = [(root, 0)]
            root_children = 0
            while stack:
                u, pi = stack[-1]
                if pi < len(adj[u]):
                    stack[-1] = (u, pi + 1)
                    v = adj[u][pi]
                    if disc[v] == -1:
                        parent[v] = u
                        if u == root:
                            root_children += 1
                        disc[v] = low[v] = timer
                        timer += 1
                        stack.append((v, 0))
                    elif v != parent[u]:
                        if disc[v] < low[u]:
                            low[u] = disc[v]
                else:
                    stack.pop()
                    p = parent[u]
                    if p != -1:
                        if low[u] < low[p]:
                            low[p] = low[u]
                        if p != root and low[u] >= disc[p]:
                            is_art[p] = True
            if root_children > 1:
                is_art[root] = True

        art_flat[ys * nelx + xs] = is_art
        return art_flat

    def _symmetric_mask(self, solid):
        """Mascara sob simetria horizontal (v22): candidato i so e valido se
        remover {i, mirror(i)} juntos preservar um unico componente conexo.
        Bruteforce por candidato (scipy.ndimage.label a cada um) — O(N)
        rotulagens por passo, mais caro que o metodo de pontos de
        articulacao, mas o cantilever (200 elementos) roda isso sem custo
        perceptivel. i == mirror(i) so ocorre com nely impar (nao usado
        neste TCC); tratado mesmo assim por completude."""
        solid_flat = solid.flatten()
        base_ok = solid_flat & ~self._protected_flat
        ok = np.zeros(self.n_elements, dtype=bool)
        for i in np.nonzero(base_ok)[0]:
            j = int(self._mirror_flat[i])
            if j != i and not base_ok[j]:
                continue
            test = solid.copy()
            test.flat[i] = False
            if j != i:
                test.flat[j] = False
            _, n = scipy.ndimage.label(test, structure=self._struct)
            if n == 1:
                ok[i] = True
        return ok

    def get_action_mask(self):
        """Hard action masking de conectividade (Huang e Ontanon, 2022; Hong
        et al., 2025), de espessura minima (ver docstring do
        modulo) e, opcionalmente, de simetria horizontal (v22): valido = solido, nao protegido, nao
        ponto de articulacao (ou par-de-espelhos nao articulador, sob
        simetria), e nao deixa vizinho abaixo da densidade local minima.
        Cacheado por estado (no maximo um calculo por passo)."""
        if not self._mask_dirty and self._mask_cache is not None:
            return self._mask_cache

        solid = self.density > 0.5
        solid_flat = solid.flatten()

        if self.simetria == "horizontal":
            ok = self._symmetric_mask(solid)
            if int(solid_flat.sum()) <= 2:
                ok[:] = False
        else:
            art_flat = self._articulation_flags(solid)
            ok = solid_flat & ~self._protected_flat & ~art_flat
            thin_flat = self._thin_flags(solid, ok)
            ok = ok & ~thin_flat
            if int(solid_flat.sum()) <= 1:
                ok[:] = False

        mask = np.zeros(self.n_elements, dtype=np.float32)
        mask[ok] = 1.0

        self._mask_cache = mask
        self._mask_dirty = False
        return mask

    def _get_action_mask_bruteforce(self):
        """Referencia O(N) por scipy.ndimage.label + convolucao recalculada
        do zero a cada candidato (mais lenta, mas obviamente correta).
        Usada apenas para verificar a equivalencia com get_action_mask() em
        07_testes_ambiente.py; nao usar em treino."""
        mask = np.ones(self.n_elements, dtype=np.float32)
        density_flat = self.density.flatten()
        for i in range(self.n_elements):
            ely = i // self.nelx
            elx = i % self.nelx
            if density_flat[i] < 0.5 or self._protected_flat[i]:
                mask[i] = 0.0
                continue
            density_test = self.density.copy()
            density_test[ely, elx] = 1e-4
            if not self._is_connected(density_test):
                mask[i] = 0.0
                continue
            if self.thin_threshold > 0:
                solid_test = density_test > 0.5
                local_test = scipy.ndimage.convolve(
                    solid_test.astype(float), self._thin_kernel,
                    mode="constant", cval=0.0)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = ely + dy, elx + dx
                    if (0 <= ny < self.nely and 0 <= nx < self.nelx
                            and solid_test[ny, nx]
                            and local_test[ny, nx] < self.thin_threshold):
                        mask[i] = 0.0
                        break
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.n_removed = 0

        # dominio inicial solido; os furos (void) nascem vazios
        self.density = np.ones((self.nely, self.nelx), dtype=np.float32)
        if self.problem_type == "automotive":
            self.density[self.void_mask] = 1e-4

        _, compliance, vm = self.solver.solve(self.density)
        self.initial_compliance = compliance
        self.current_compliance = compliance
        self.vm_map = vm

        if self.dfam_peso > 0.0:
            self._viol_prev = self._violacao_overhang(self.density > 0.5)

        self._mask_dirty = True
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        ely = action // self.nelx
        elx = action % self.nelx

        # rede de seguranca: a mascara deve impedir chegar aqui com acao invalida.
        # Se acontecer, e falha do wrapper; penaliza e encerra.
        if self.density[ely, elx] < 0.5 or self.protected_mask[ely, elx]:
            return self._get_obs(), -self.deadlock_penalty, True, False, {
                "compliance": self.current_compliance,
                "volume": float(np.mean(self.density)),
                "n_removed": self.n_removed,
                "invalid_action": True,
                "deadlock": False,
            }

        # remocao valida (a mascara ja garantiu conectividade): E_e de 1 para 1e-4
        self.density[ely, elx] = 1e-4
        self.n_removed += 1
        if self.simetria == "horizontal":
            # remove tambem o espelho na MESMA transicao (um so solve de MEF
            # para o par) — a mascara ja garantiu que o par preserva
            # conectividade. j == action so com nely impar (nao usado aqui).
            j = int(self._mirror_flat[action])
            if j != action:
                jy, jx = j // self.nelx, j % self.nelx
                self.density[jy, jx] = 1e-4
                self.n_removed += 1
        self._mask_dirty = True

        try:
            _, new_compliance, vm = self.solver.solve(self.density)
            self.vm_map = vm
        except Exception:
            return self._get_obs(), -self.deadlock_penalty, True, False, {
                "compliance": float("inf"),
                "volume": float(np.mean(self.density)),
                "n_removed": self.n_removed,
                "solver_failed": True,
                "deadlock": False,
            }

        # recompensa (ver item 2 da docstring do modulo)
        c_prev = max(float(self.current_compliance), 1e-12)
        c_t = max(float(new_compliance), 1e-12)
        if self.reward_mode == "brown":
            # forma original de Brown et al. (2022), Eq. 4 — mantida para
            # reproduzir as rodadas v19/v20
            reward = ((self.initial_compliance / c_t) ** 2
                      + (self.n_removed / self.n_elements) ** 2)
        else:
            # telescopica: a soma no episodio = reward_scale*log(c_0/c_T)
            reward = self.reward_scale * float(np.log(c_prev / c_t))
            if not np.isfinite(reward):
                reward = -self.deadlock_penalty

        if self.dfam_peso > 0.0:
            viol = self._violacao_overhang(self.density > 0.5)
            reward -= self.dfam_peso * (viol - self._viol_prev)
            self._viol_prev = viol

        # parada ao atingir a fracao de volume alvo
        done = self.n_removed >= self.n_to_remove
        deadlock = False
        if not done:
            # terminacao por deadlock: se nenhuma acao valida restar antes do VF
            # alvo, encerra de forma limpa em vez de deixar o MaskablePPO
            # amostrar de uma distribuicao vazia (NaN). A mascara calculada
            # aqui fica em cache e e reaproveitada pelo wrapper no proximo
            # estado, sem custo extra.
            if float(self.get_action_mask().sum()) == 0.0:
                done = True
                deadlock = True
                if self.reward_mode != "brown":
                    # o VF alvo nao foi atingido: projeto inviavel. Penaliza
                    # proporcionalmente as remocoes que faltaram, em vez de
                    # deixar o episodio curto parecer barato.
                    faltam = max(self.n_to_remove - self.n_removed, 0)
                    reward -= (self.deadlock_penalty
                               * faltam / max(self.n_to_remove, 1))

        self.current_compliance = new_compliance

        info = {
            "compliance": new_compliance,
            "volume": float(np.mean(self.density)),
            "n_removed": self.n_removed,
            "invalid_action": False,
            "deadlock": deadlock,
        }
        if self.dfam_peso > 0.0:
            info["dfam_violacao"] = self._viol_prev
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """Cinco canais; ver item 1 da docstring do modulo.

        c0 densidade binaria
        c1 tensao normalizada pelo percentil 99 dos solidos, clipada [0,1]
        c2 posto normalizado da tensao entre os solidos, em [0,1]
           (argmin de c2 entre as acoes validas = politica gulosa ESO)
        c3 elementos com no engastado
        c4 elemento carregado
        """
        solid = self.density > 0.5
        c0 = solid.astype(np.float32)

        vm = np.where(solid, np.nan_to_num(self.vm_map, nan=0.0,
                                           posinf=0.0, neginf=0.0), 0.0)
        vals = vm[solid]
        ref = float(np.percentile(vals, 99.0)) if vals.size else 0.0
        if not np.isfinite(ref) or ref <= 1e-12:
            ref = 1.0
        c1 = np.clip(vm / ref, 0.0, 1.0).astype(np.float32)

        c2 = np.zeros(self.n_elements, dtype=np.float32)
        idx = np.nonzero(solid.ravel())[0]
        if idx.size:
            ordem = np.argsort(vm.ravel()[idx], kind="stable")
            postos = np.empty(idx.size, dtype=np.float32)
            postos[ordem] = (np.arange(idx.size, dtype=np.float32)
                             / max(idx.size - 1, 1))
            c2[idx] = postos
        c2 = c2.reshape(self.nely, self.nelx)

        return np.stack([c0, c1, c2,
                         self._fixed_channel, self._load_channel], axis=0)



def nucleo_conico(raio):
    """Mesmo nucleo w(d) = max(0, rmin - d) de precompute_filter(),
    normalizado para somar 1."""
    r = int(np.ceil(raio))
    yy, xx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    k = np.maximum(0.0, raio - np.sqrt(xx ** 2 + yy ** 2))
    return k / k.sum()


def rodar_guloso(nelx=30, nely=30, vol_frac=0.3, problem_type="automotive",
                 raio_filtro=1.5, seed=0, thin_threshold=0.0, simetria=None,
                 verbose=True):
    """Episodio guloso: remove sempre o elemento valido de menor tensao (sob
    simetria horizontal, o par elemento+espelho de menor tensao combinada;
    ver docstring de este modulo, v22)."""
    env = TopoOptMaskEnv(nelx=nelx, nely=nely, vol_frac=vol_frac,
                         problem_type=problem_type,
                         thin_threshold=thin_threshold, simetria=simetria)
    env.reset(seed=seed)
    K = nucleo_conico(raio_filtro) if raio_filtro > 0 else None

    t0 = time.time()
    done, info = False, {}
    while not done:
        validas = np.nonzero(env.get_action_mask() > 0)[0]
        if validas.size == 0:
            break
        campo = env.vm_map
        if K is not None:
            campo = scipy.ndimage.convolve(campo, K, mode="nearest")
        campo_flat = campo.flatten()
        if simetria == "horizontal":
            # custo do PAR (elemento + espelho), nao so do elemento: a acao
            # sempre remove os dois juntos, entao o criterio guloso certo e
            # a tensao combinada do par, nao a do elemento isolado.
            custo = campo_flat[validas] + campo_flat[env._mirror_flat[validas]]
        else:
            custo = campo_flat[validas]
        acao = int(validas[np.argmin(custo)])
        _, _, term, trunc, info = env.step(acao)
        done = term or trunc

    dt = time.time() - t0
    if verbose:
        tag = f"filtro rmin={raio_filtro}" if raio_filtro > 0 else "sem filtro"
        print(f"ESO guloso ({tag}): C = {info['compliance']:.4f} | "
              f"{env.n_removed}/{env.n_to_remove} remocoes | "
              f"deadlock={info.get('deadlock', False)} | {dt:.1f}s | "
              f"{env.n_removed} chamadas de MEF")
    return env.density.copy(), float(info["compliance"]), env.n_removed, dt


def topo_mask_fn(env):
    return env.get_action_mask()
