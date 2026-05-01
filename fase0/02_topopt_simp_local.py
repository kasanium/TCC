import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
# import pandas as pd # comentado pq decidi nao usar pra exportar dados agora
import matplotlib.pyplot as plt

# config global de figuras
# ---------------------------------------------------------
# TODO: ver se essa fonte tem em todas as maquinas depois
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.linewidth": 1.5,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

def formatar_e_salvar(fig, ax, titulo, caminho_saida, is_image=True):
    # formata os plots. se for is_image=True ele limpa os eixos (pra matriz de topologia)
    ax.set_title(titulo, pad=15)

    if is_image:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig(caminho_saida)
    plt.close(fig)
    print(f"Arquivo salvo em: {caminho_saida}")

def lk():
    # matriz de rigidez do elemento (valores baseados no paper do andreassen 2012)
    E, nu = 1.0, 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])

    return E/(1-nu**2)*np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

def precompute_filter(nelx, nely, rmin):
    # pre-calcula a matriz do filtro H
    # FIXME: se a malha for mt grande, essa matriz densa vai estourar a memoria.
    # pensar em usar scipy.sparse depois

    if rmin <= 0:
        print("aviso: rmin zerado ou negativo, filtro inativo na pratica")

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

def FE(nelx, nely, x, penal):
    KE = lk()
    e0 = 1.0 
    emin = 1e-9 # limite da eq 1 do andreassen pra nao zerar K
    
    ndof = 2*(nelx+1)*(nely+1)
    K = np.zeros((ndof, ndof))
    F = np.zeros((ndof, 1))
    U = np.zeros((ndof, 1))
    
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            
            # dofs do elemento (repetitivo mesmo)
            edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
            idx = ely + elx*nely
            
            # eq 1: E(x) = emin + x^p * (e0 - emin)
            E_val = emin + (x[idx]**penal) * (e0 - emin)
            K[np.ix_(edof, edof)] += E_val * KE
            
    # sec 5.1 andreassen: carga pontual no finalzinho (cantilever)
    F[ndof - 1, 0] = -1.0 
    
    # fixando parede esquerda (x=0)
    fixeddofs = np.arange(2*(nely+1))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    
    # resolve o sistema
    if len(freedofs) > 0:
        # np.linalg.solve eh chatinho se a matriz for singular, mas o emin salva a gente
        U[freedofs, 0] = np.linalg.solve(K[np.ix_(freedofs, freedofs)], F[freedofs, 0])
        
    return U

def sensitivity_analysis(nelx, nely, x, penal, U):
    KE = lk()
    c = 0.0
    dc = np.zeros(nely * nelx)
    e0, emin = 1.0, 1e-9
    
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3])
            
            Ue = U[edof, 0]
            ce = Ue.T @ KE @ Ue # energia mutua / compliance do elemento
            idx = ely + elx*nely
            
            c += (emin + (x[idx]**penal) * (e0 - emin)) * ce
            
            # eq 5 do paper: derivada
            dc[idx] = -penal * (x[idx]**(penal-1)) * (e0 - emin) * ce
            
    return c, dc

def optimality_criteria_update(nelx, nely, x, volfrac, dc, H, Hs):
    # filtro das sensibilidades - eq 7 do andreassen
    # max(1e-3, x) pra evitar divisao por 0
    dc = (H @ (x * dc)) / Hs / np.maximum(1e-3, x)
    
    l1, l2 = 0.0, 100000.0
    move = 0.2 # sigmund usa esse default
    
    # bisseccao
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        
        # update chato cheio de max/min da eq 3
        # ps: arrumei pra usar sqrt(-dc/lmid) q nem no script do matlab
        xnew = np.maximum(0.001, 
               np.maximum(x - move, 
               np.minimum(1.0, 
               np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        
        if xnew.mean() > volfrac: 
            l1 = lmid
        else: 
            l2 = lmid
            
    return xnew

if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _OUT = os.path.join(_HERE, 'resultados', '02_baseline.png')
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)

    nelx, nely = 32, 20
    volfrac = 0.4
    penal = 3.0
    rmin = 1.2
    
    x = volfrac * np.ones(nely * nelx, dtype=float)
    
    loop = 0
    change = 1.0
    
    H, Hs = precompute_filter(nelx, nely, rmin)

    # limitando a 100 pq demora mt no cpu se deixar rolar
    while change > 0.01 and loop < 100:
        loop += 1
        xold = x.copy()
        
        U = FE(nelx, nely, x, penal)
        c, dc = sensitivity_analysis(nelx, nely, x, penal, U)
        x = optimality_criteria_update(nelx, nely, x, volfrac, dc, H, Hs)
        
        # l-inf norm (equivalente ao max abs diff)
        change = np.linalg.norm(x - xold, np.inf)
        
        if loop % 5 == 0 or change <= 0.01:
            print(f"iter: {loop:3d} | obj: {c:6.3f} | vol: {x.mean():.3f} | mudanca: {change:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    # reshape e transpõe pra ficar na orientacao certa
    ax.imshow(x.reshape((nelx, nely)).T, cmap='gray_r', interpolation='none', aspect='equal')
    
    label_titulo = f'SIMP - Cantilever {nelx}x{nely} | C = {c:.2f}'
    formatar_e_salvar(fig, ax, label_titulo, _OUT, is_image=True)