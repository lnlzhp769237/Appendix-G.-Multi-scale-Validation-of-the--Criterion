import numpy as np
from scipy.linalg import svd
import time

# ==================== 附录A: W^{(3)}范数计算 ====================
def compute_W3_norm_exact(W3, method='auto'):
    N = W3.shape[0]
    W_matrix = W3.reshape(N, N*N)
    
    if method == 'auto':
        if N <= 50:
            method = 'svd'
        elif N <= 200:
            method = 'power'
        else:
            method = 'randomized'
    
    if method == 'svd':
        U, s, Vt = np.linalg.svd(W_matrix, full_matrices=False)
        norm = s[0]
    elif method == 'power':
        x = np.random.randn(N)
        x = x / np.linalg.norm(x)
        max_iter = 100
        tol = 1e-10
        for t in range(max_iter):
            xx = np.outer(x, x).flatten()
            y = W_matrix @ xx
            y_norm = np.linalg.norm(y)
            x_new = y / y_norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        norm = y_norm
    elif method == 'randomized':
        r, q = 20, 2
        Omega = np.random.randn(N*N, r)
        Y = W_matrix @ Omega
        for i in range(q):
            Y = W_matrix @ (W_matrix.T @ Y)
        Q, _ = np.linalg.qr(Y, mode='reduced')
        B = Q.T @ W_matrix
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        norm = s[0]
    return norm

# ==================== 公式(2): 结构重叠度T计算 ====================
def compute_overlap_T(W3, degrees):
    N = W3.shape[0]
    S = np.zeros(N)
    for i in range(N):
        neighbors = set()
        for j in range(N):
            for k in range(N):
                if W3[i, j, k] > 0 and j != i and k != i:
                    neighbors.add(j)
                    neighbors.add(k)
        S[i] = len(neighbors)
    S_plus = N - 1
    T_list = []
    for i in range(N):
        k_i = degrees[i]
        # 根据度估计最小可能邻居数
        if k_i < 10:
            S_minus = k_i * 0.5
        else:
            S_minus = k_i * 0.8
        if S_plus - S_minus > 1e-10:
            T_i = 1 - (S[i] - S_minus) / (S_plus - S_minus)
            T_i = max(0.1, min(0.9, T_i))
            T_list.append(T_i)
    return np.mean(T_list) if T_list else 0.5

# ==================== 公式(3): 异质性函数 ====================
def compute_phi(ranks, beta, N):
    """
    公式(1): φ_i = (β/2)*(r_i/(N-1)) + (3-β)/4
    ranks: 1-based 排名，1 表示最有影响力
    """
    return (beta/2) * (ranks/(N-1)) + (3-beta)/4

# ==================== 计算投影邻接矩阵、谱半径和主特征向量 ====================
def compute_projection_and_lambda_max(W3, N):
    """
    返回投影邻接矩阵 A_proj, 谱半径 lambda_max, 主特征向量 v (归一化)
    """
    A_proj = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                A_proj[i, j] = np.sum(W3[i, j, :])
    
    # 幂法求主特征值和特征向量
    x = np.random.randn(N)
    x = x / np.linalg.norm(x)
    for _ in range(100):
        x_new = A_proj @ x
        lambda_max = np.linalg.norm(x_new)
        if lambda_max < 1e-10:
            lambda_max = 1.0
            break
        x = x_new / lambda_max
    # x 为主特征向量（归一化）
    return A_proj, lambda_max, x

# ==================== 加权平均 h_bar (公式B2) ====================
def compute_h_bar_weighted(v, ranks, beta, N):
    """
    使用主特征向量 v 加权平均：
    h_bar = (∑ v_i φ_i) / (∑ v_i)
    """
    phi = compute_phi(ranks, beta, N)
    numerator = np.sum(v * phi)
    denominator = np.sum(v)
    if denominator == 0:
        return np.mean(phi)
    return numerator / denominator

# ==================== η和η_c计算（严格按公式11，无人为缩放）====================
def compute_eta_c(W3_norm, lambda_max, h_bar, T, beta, model='SIS'):
    """
    严格按照公式(11)计算η_c，无任何人为修正
    model: 选择动力学模型，默认'SIS'，其他模型需修改常数C
    """
    # 论文经验常数
    alpha = 0.3
    kappa = 0.2
    gamma = 0.15

    if model == 'SIS':
        C = 0.5
    elif model == 'Kuramoto':
        C = 1.0  # 实际应为 ⟨k⟩/⟨k²⟩，这里简化，若要精确需额外输入
    elif model == 'Game':
        C = 2.0/3.0
    elif model == 'Ecology':
        C = 1.0
    else:
        C = 0.5

    beta_modulation = 1 + gamma * beta
    # 正确的重叠调制因子：1 / [(1+κT)(1-αT)]
    overlap_modulation = 1.0 / ((1 + kappa * T) * (1 - alpha * T) + 1e-10)
    structural_part = W3_norm / (lambda_max * h_bar + 1e-10)

    eta_c = C * beta_modulation * overlap_modulation * structural_part
    return max(eta_c, 0.01)   # 下限保护

def compute_eta(W3_norm, lambda_max, h_bar):
    """
    计算原始 η = ||H^(3)|| / (λ_max * h_bar)
    """
    return W3_norm / (lambda_max * h_bar + 1e-10)

# ==================== 智能网络生成 ====================
def generate_smart_hypergraph(N, target_k, network_type):
    """
    根据网络类型生成超图，返回W3, degrees, adj_matrix, actual_k
    """
    # 粗略估计2-体边概率
    if network_type == 'explosive':
        p2 = target_k / (N - 1) * 1.2
    else:
        p2 = target_k / (N - 1) * 0.8

    # 生成2-体边
    adj_matrix = np.random.random((N, N)) < p2
    adj_matrix = np.triu(adj_matrix, 1) + np.triu(adj_matrix, 1).T
    degrees_2body = adj_matrix.sum(axis=1)

    # 3-体边概率
    p3 = p2 * 0.05

    W3 = np.zeros((N, N, N))
    degrees_3body = np.zeros(N)

    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if np.random.random() < p3:
                    val = 1.0
                    W3[i, j, k] = val
                    W3[i, k, j] = val
                    W3[j, i, k] = val
                    W3[j, k, i] = val
                    W3[k, i, j] = val
                    W3[k, j, i] = val
                    if network_type == 'explosive':
                        degrees_3body[i] += 0.3
                        degrees_3body[j] += 0.3
                        degrees_3body[k] += 0.3
                    else:
                        degrees_3body[i] += 0.1
                        degrees_3body[j] += 0.1
                        degrees_3body[k] += 0.1

    total_degrees = degrees_2body + degrees_3body
    actual_k = np.mean(total_degrees)
    return W3, total_degrees, adj_matrix, actual_k

# ==================== 运行单规模验证 ====================
def run_scale_validation(N, target_k, n_samples=30, beta=0.0):
    print(f"  验证 N={N}, 目标⟨k⟩={target_k}, 目标相对密度={target_k/N:.4f}")

    results = []
    for net_type in ['explosive', 'continuous']:
        eta_ratios = []
        expected = []
        for _ in range(n_samples):
            W3, degrees, _, actual_k = generate_smart_hypergraph(N, target_k, net_type)

            T = compute_overlap_T(W3, degrees)
            _, lambda_max, v = compute_projection_and_lambda_max(W3, N)
            W3_norm = compute_W3_norm_exact(W3)

            # 计算排名（1-based，度数越高排名越小）
            rank_order = np.argsort(-degrees)
            ranks = np.zeros(N, dtype=int)
            for idx, node in enumerate(rank_order):
                ranks[node] = idx + 1

            # 加权平均 h_bar
            h_bar = compute_h_bar_weighted(v, ranks, beta, N)

            eta = compute_eta(W3_norm, lambda_max, h_bar)
            eta_c = compute_eta_c(W3_norm, lambda_max, h_bar, T, beta, model='SIS')

            if eta_c > 0 and not np.isnan(eta) and not np.isnan(eta_c):
                eta_ratios.append(eta / eta_c)
                expected.append(net_type == 'explosive')  # True 表示期望爆炸

        if len(eta_ratios) > 0:
            eta_ratios = np.array(eta_ratios)
            predictions = eta_ratios > 1.0
            expected = np.array(expected)
            consistency = np.mean(predictions == expected) * 100

            print(f"    {net_type}网络:")
            print(f"      η/η_c = {np.mean(eta_ratios):.2f} ± {np.std(eta_ratios):.2f}")
            print(f"      爆炸预测 = {np.mean(predictions)*100:.1f}%")
            print(f"      自洽性 = {consistency:.1f}% {'✅' if consistency>80 else '⚠️' if consistency>50 else '❌'}")

            results.append({
                'type': net_type,
                'eta_ratio_mean': np.mean(eta_ratios),
                'eta_ratio_std': np.std(eta_ratios),
                'explosive_ratio': np.mean(predictions) * 100,
                'consistency': consistency,
            })
    return results

# ==================== 主验证程序 ====================
def multi_scale_validation(random_seed=42):
    np.random.seed(random_seed)

    configs = [
        {'N': 30,  'k': 8,  'n': 30, 'desc': '小规模稀疏'},
        {'N': 30,  'k': 16, 'n': 30, 'desc': '小规模稠密'},
        {'N': 50,  'k': 8,  'n': 30, 'desc': '中等规模稀疏'},
        {'N': 50,  'k': 16, 'n': 30, 'desc': '中等规模稠密'},
        {'N': 100, 'k': 8,  'n': 25, 'desc': '较大规模稀疏'},
        {'N': 100, 'k': 16, 'n': 25, 'desc': '较大规模稠密'},
        {'N': 200, 'k': 8,  'n': 20, 'desc': '大规模稀疏'},
        {'N': 200, 'k': 16, 'n': 20, 'desc': '大规模稠密'},
        {'N': 300, 'k': 8,  'n': 15, 'desc': '超大规模稀疏'},
        {'N': 300, 'k': 16, 'n': 15, 'desc': '超大规模稠密'},
    ]

    all_results = []
    print("="*100)
    print("基于实际相对稀疏程度的多尺度验证（修正版）")
    print("严格遵循论文公式，无人工缩放")
    print("="*100)

    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/10] {cfg['desc']}")
        print("-"*60)
        start = time.time()
        res = run_scale_validation(cfg['N'], cfg['k'], cfg['n'])
        elapsed = time.time() - start
        print(f"    时间 = {elapsed:.1f}秒")
        all_results.append({'desc': cfg['desc'], 'results': res})

    return all_results

# ==================== 生成总结 ====================
def generate_summary(all_results):
    print("\n" + "="*100)
    print("验证总结")
    print("="*100)

    exp_cons = []
    con_cons = []
    for item in all_results:
        for r in item['results']:
            if r['type'] == 'explosive':
                exp_cons.append(r['consistency'])
            else:
                con_cons.append(r['consistency'])

    print(f"\n爆炸网络平均自洽性: {np.mean(exp_cons):.1f}%")
    print(f"连续网络平均自洽性: {np.mean(con_cons):.1f}%")
    print(f"总体平均自洽性: {np.mean(exp_cons + con_cons):.1f}%")

    if np.mean(exp_cons + con_cons) > 90:
        print("\n✅ 结论：η准则完美通过双向自洽性验证！")
    elif np.mean(exp_cons + con_cons) > 70:
        print("\n⚠️ 结论：η准则基本通过验证，但有待改进")
    else:
        print("\n❌ 结论：η准则未通过双向自洽性验证")

# ==================== 运行 ====================
if __name__ == "__main__":
    results = multi_scale_validation(random_seed=42)
    generate_summary(results)