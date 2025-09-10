# --- nsga_core.py ---
from dataclasses import dataclass
import random, math, copy
from typing import List, Tuple, Dict
from simulator.builder import ModelBuilder
from simulator.engine.simulator import Simulator
from simulator.result.recorder import Recorder
import os, csv
import matplotlib.pyplot as plt


def hv_2d(fits, ref=(1e6, 1e6)):
    """2목적 최소화 하이퍼볼륨 (참조점 ref 기준, 작은 값일수록 좋음).
       fits: [(f1,f2), ...] (모두 유한)"""
    if not fits: return 0.0
    # 비지배해만 사용 (이미 F1만 넘겨도 됨). f1 오름차순으로 정렬
    pts = sorted(fits, key=lambda x: (x[0], x[1]))
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in pts:
        width  = max(0.0, ref[0] - f1)
        height = max(0.0, prev_f2 - f2)
        hv += width * height
        prev_f2 = f2
    return hv

def spread_delta(front_fits):
    """NSGA-II diversity metric Δ
       Δ = (d_f + d_l + sum|d_i - d̄|) / (d_f + d_l + (|F|-1)*d̄)"""
    n = len(front_fits)
    if n <= 2:
        return 0.0
    pts = sorted(front_fits, key=lambda x: (x[0], x[1]))
    # 유클리드 인접 거리들
    d = [math.dist(pts[i], pts[i+1]) for i in range(n-1)]
    dbar = sum(d)/len(d)
    d_f = d[0]
    d_l = d[-1]
    num = d_f + d_l + sum(abs(di - dbar) for di in d)
    den = d_f + d_l + (len(d))*dbar
    return num/den if den > 0 else 0.0

def save_metrics_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)

def plot_curves(metrics_csv, out_png):
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        fig = plt.figure(figsize=(6,4))
        plt.plot(df["gen"], df["hv"], label="Hypervolume")
        plt.plot(df["gen"], df["best_f1"], label="min f1")
        plt.plot(df["gen"], df["best_f2"], label="min f2")
        plt.plot(df["gen"], df["spread"], label="Δ (spread)")
        plt.xlabel("Generation"); plt.legend(); plt.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plot_curves skipped: {e}")

# ===== 0) 유틸: 시나리오에서 글로벌 operation 인덱스 생성 =====
def build_operation_index(scenario_path: str):
    """operations.json과 jobs.json을 통해 '전역 공정 토큰 리스트'와
    'op_id -> 후보머신 리스트' 매핑을 만든다."""
    # ModelBuilder를 통해 파싱(객체까지 만들 필요는 없어도, builder가 경로 맞춰줌)
    mb = ModelBuilder(scenario_path, use_dynamic_scheduling=True)  # 동적 모드
    # 내부 json 직접 읽기 위해 builder의 path 사용
    path = mb.path  # builder가 절대경로로 바꿔둠  :contentReference[oaicite:4]{index=4}
    import json, os
    with open(os.path.join(path, "jobs.json"), "r", encoding="utf-8") as f:
        jobs_j = json.load(f)
    with open(os.path.join(path, "operations.json"), "r", encoding="utf-8") as f:
        ops_j = json.load(f)

    op_map = {o["operation_id"]: o for o in ops_j}
    op_tokens = []    # 전역 공정 토큰 시퀀스 (예: ["J1-O1","J1-O2","J2-O1",...])
    op_to_candidates = {}
    for j in jobs_j:
        for oid in j["operations"]:
            op_tokens.append(oid)
            op_to_candidates[oid] = op_map[oid]["machines"]

    return op_tokens, op_to_candidates

# ===== 1) 염색체 =====
@dataclass
class Chromosome:
    op_seq: List[str]          # 전역 공정 순서 (operation_id 리스트)
    mac_seq: List[str]         # 각 op에 할당할 머신 이름
    agv_seq: List[str]         # 각 op에 할당할 AGV id (선택적; 현재는 placeholder)

# ===== 2) 초기해 생성 =====
def init_population(pop_size: int, op_tokens: List[str], op_to_candidates: Dict[str, List[str]]) -> List[Chromosome]:
    pop = []
    for _ in range(pop_size):
        # (a) operation 순서: 같은 job 내 선행제약만 지키며 셔플해야 하지만,
        #    초기 버전은 "jobs.json"의 기본 순서를 유지하고 전체를 살짝 섞는 정도로 시작
        op_seq = op_tokens[:]
        random.shuffle(op_seq)

        # (b) 머신층: 각 op마다 후보에서 랜덤
        mac_seq = []
        for oid in op_seq:
            cands = op_to_candidates.get(oid, [])
            mac_seq.append(random.choice(cands) if cands else None)

        # (c) AGV층: 아직 단일-머신-AGV 구조가 명확하지 않아 placeholder
        agv_seq = ["AGV_1" for _ in op_seq]  # 필요 시 확장

        pop.append(Chromosome(op_seq, mac_seq, agv_seq))
    return pop

def init_population_partial(pop_size: int,
                            sel_ops: List[str],
                            op_to_candidates: Dict[str, List[str]]) -> List[Chromosome]:
    pop = []
    for _ in range(pop_size):
        # op_seq: 선택된 op만 섞음 (선행제약은 시뮬레이터가 실행시점에 보장하므로 여기선 단순 셔플)
        op_seq = sel_ops[:]
        random.shuffle(op_seq)

        # mac_seq: 각 op별 후보 중 랜덤
        mac_seq = []
        for oid in op_seq:
            cands = op_to_candidates.get(oid, [])
            mac_seq.append(random.choice(cands) if cands else None)

        agv_seq = ["AGV_1" for _ in op_seq]  # placeholder
        pop.append(Chromosome(op_seq, mac_seq, agv_seq))
    return pop

def sample_ops_for_nsga(scenario_path: str, n_max: int, seed: int = 0):
    """전역 op 리스트에서 무작위로 n_max개만 뽑아 부분 인코딩 대상으로 사용"""
    op_tokens, op_to_candidates = build_operation_index(scenario_path)
    n = min(max(int(n_max), 0), len(op_tokens))
    random.seed(seed)
    selected = random.sample(op_tokens, n) if n > 0 else []
    # 후보 사전도 부분 집합으로 슬라이스
    sel_cands = {oid: op_to_candidates.get(oid, []) for oid in selected}
    return selected, sel_cands

# ===== 3) 평가: 시뮬레이터를 "콜백"으로 구동 =====
def evaluate(ch: Chromosome, scenario_path: str) -> Tuple[float, float]:
    """
    반환: (f1, f2) = (makespan, total_agv_time)
    - makespan은 보통 'run' 종료 시 Simulator의 current_time으로 잡는다.
    - total_agv_time은 머신의 agv_logs에서 delivery/return 합으로 계산.
    """
    from simulator.result.recorder import Recorder
    Recorder.enabled = False   # 평가할 때는 기록 끔
    Recorder.records.clear()   # 혹시 남은 기록 있으면 비움
    # (1) 모델 구성
    builder = ModelBuilder(scenario_path, use_dynamic_scheduling=True)
    machines, gen, tx, agv_controller, agvs = builder.build()  # Job/Operation은 동적 할당 모드  :contentReference[oaicite:5]{index=5}

    # (2) 시뮬레이터 생성 및 등록
    sim = Simulator()
    for m in machines:
        m.simulator = sim
        sim.register(m)
    sim.register(gen)
    sim.register(tx)

    sim.register(agv_controller)
    for agv in agvs:
        sim.register(agv)

    # (3) "선택 콜백" 주입: op_id -> 머신 이름 매핑을 빠르게 조회하도록 dict 구성
    # ch.op_seq와 ch.mac_seq는 같은 길이이며, 같은 위치의 op에 대한 머신 배정
    plan: Dict[str, str] = {op_id: mac for op_id, mac in zip(ch.op_seq, ch.mac_seq)}

    def select_next_machine(job_id: str, op_id: str, candidates: List[str]):
        # 염색체에 지정된 머신이 후보에 없으면, 후보[0]로 폴백
        sel = plan.get(op_id)
        if sel in (candidates or []):
            return sel
        return candidates[0] if candidates else None

    sim.select_next_machine = select_next_machine  # ✅ 1)에서 추가한 훅 사용

    # (4) Generator 초기화 및 실행
    if hasattr(gen, "initialize"):
        gen.initialize()

    sim.run()  # main과 동일한 호출(인자 생략 가능)  :contentReference[oaicite:6]{index=6}

    # (5) 목적함수 계산
    # f1: makespan -> 시뮬레이터의 현재 시간(마지막 이벤트 시각)으로 가정
    makespan = getattr(sim, "current_time", None)
    if makespan is None:
        # 혹시 엔진 구현에 따라 없으면 Recorder 등에서 최대 완료시간 계산(백업)
        makespan = Recorder.get_makespan() if hasattr(Recorder, "get_makespan") else float("inf")

    # f2: 총 AGV 이동시간 -> 각 Machine.agv_logs에서 delivery/return 합산
    total_agv_time = 0.0
    for m in machines:
        if hasattr(m, "agv_logs"):
            for rec in m.agv_logs:
                if rec["activity_type"] in ("delivery_start", "return_home"):
                    total_agv_time += float(rec.get("duration", 0.0))

    return float(makespan), float(total_agv_time)

def evaluate_partial(ch: Chromosome,
                     scenario_path: str,
                     selected_ops: List[str]) -> Tuple[float, float]:
    from simulator.result.recorder import Recorder
    Recorder.enabled = False
    Recorder.records.clear()

    # 1) 모델 구성 (동적 라우팅 모드)
    builder = ModelBuilder(scenario_path, use_dynamic_scheduling=True)
    machines, gen, tx, agv_controller, agvs = builder.build()

    # 2) 시뮬레이터 등록
    sim = Simulator()
    for m in machines:
        m.simulator = sim
        sim.register(m)
    sim.register(gen)
    sim.register(tx)
    sim.register(agv_controller)
    for agv in agvs:
        sim.register(agv)

    # 3) plan: 염색체에 포함된 (선택된) op -> machine
    plan: Dict[str, str] = {op_id: mac for op_id, mac in zip(ch.op_seq, ch.mac_seq)}
    selset = set(selected_ops)

    # [룰기반] 후보[0] 폴백 (원하면 큐부하 최소 등으로 교체 가능)
    def rule_next_machine(job_id: str, op_id: str, candidates: List[str]):
        return candidates[0] if candidates else None

    # 하이브리드 선택 훅
    def select_next_machine(job_id: str, op_id: str, candidates: List[str]):
        if op_id in selset:
            m = plan.get(op_id)
            if m in (candidates or []):
                return m
            # 계획-후보 불일치 시 안전 폴백
            return candidates[0] if candidates else None
        # 비선택 op는 룰기반
        return rule_next_machine(job_id, op_id, candidates)

    sim.select_next_machine = select_next_machine

    # 4) 실행
    if hasattr(gen, "initialize"):
        gen.initialize()
    sim.run()

    # 5) 목적함수 (makespan, total_agv_time)
    makespan = getattr(sim, "current_time", None)
    if makespan is None and hasattr(Recorder, "get_makespan"):
        makespan = Recorder.get_makespan()
    if makespan is None:
        makespan = float("inf")

    total_agv_time = 0.0
    for m in machines:
        if hasattr(m, "agv_logs"):
            for rec in m.agv_logs:
                if rec["activity_type"] in ("delivery_start", "return_home"):
                    total_agv_time += float(rec.get("duration", 0.0))

    return float(makespan), float(total_agv_time)

# ===== 4) NSGA-II 핵심(아주 간결 버전) =====
def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a != b)

def nondominated_sort(fits: List[Tuple[float, float]]) -> List[List[int]]:
    S = [[] for _ in fits]
    n = [0]*len(fits)
    fronts = [[]]
    for p in range(len(fits)):
        for q in range(len(fits)):
            if dominates(fits[p], fits[q]):
                S[p].append(q)
            elif dominates(fits[q], fits[p]):
                n[p] += 1
        if n[p]==0:
            fronts[0].append(p)
    i=0
    while fronts[i]:
        nxt=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0:
                    nxt.append(q)
        i+=1
        fronts.append(nxt)
    return fronts[:-1]

def crowding_distance(front: List[int], fits: List[Tuple[float, float]]) -> Dict[int, float]:
    if not front: return {}
    l = len(front)
    dist = {i:0.0 for i in front}
    for m in range(2):  # f1, f2
        front_sorted = sorted(front, key=lambda i: fits[i][m])
        dist[front_sorted[0]] = dist[front_sorted[-1]] = float('inf')
        fmin, fmax = fits[front_sorted[0]][m], fits[front_sorted[-1]][m]
        denom = (fmax - fmin) or 1.0
        for k in range(1, l-1):
            i_prev, i_next = front_sorted[k-1], front_sorted[k+1]
            dist[front_sorted[k]] += (fits[i_next][m] - fits[i_prev][m]) / denom
    return dist

def tournament_select(indices: List[int], fits, fronts, cd, k=2):
    best = None
    for _ in range(k):
        i = random.choice(indices)
        key = (next(fi for fi,f in enumerate(fronts) if i in f), -cd.get(i,0.0))
        if (best is None) or (key < best[0]):
            best = (key, i)
    return best[1]

def crossover(p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    # 간단: 머신/AGV층은 uniform, op_seq는 order-crossover(OX) 대용으로 위치 유지 섞기
    def ox(a, b):
        L=len(a); i,j=sorted(random.sample(range(L),2))
        mid=a[i:j]
        tail=[x for x in b if x not in mid]
        return tail[:i]+mid+tail[i:]
    c1_seq = ox(p1.op_seq, p2.op_seq)
    c2_seq = ox(p2.op_seq, p1.op_seq)
    def uni(a,b): return [random.choice([x,y]) for x,y in zip(a,b)]
    c1 = Chromosome(c1_seq, uni(p1.mac_seq,p2.mac_seq), uni(p1.agv_seq,p2.agv_seq))
    c2 = Chromosome(c2_seq, uni(p1.mac_seq,p2.mac_seq), uni(p1.agv_seq,p2.agv_seq))
    return c1, c2

def mutate(ind: Chromosome, op_to_candidates, p=0.1):
    # op_seq swap
    if random.random()<p:
        i,j=sorted(random.sample(range(len(ind.op_seq)),2))
        ind.op_seq[i], ind.op_seq[j] = ind.op_seq[j], ind.op_seq[i]
    # mac_seq 재샘플
    for t,oid in enumerate(ind.op_seq):
        if random.random()<p:
            cands = op_to_candidates.get(oid, [])
            if cands: ind.mac_seq[t] = random.choice(cands)

def repair_partial(ch: Chromosome, op_to_candidates: Dict[str, List[str]]) -> Chromosome:
    fixed = False
    mac_seq = ch.mac_seq[:]
    for i, oid in enumerate(ch.op_seq):
        cands = op_to_candidates.get(oid, [])
        if not cands:
            continue
        if mac_seq[i] not in cands:
            mac_seq[i] = random.choice(cands)
            fixed = True
    if fixed:
        return Chromosome(ch.op_seq[:], mac_seq, ch.agv_seq[:])
    return ch

def nsga2_run(scenario_path: str, pop_size=50, generations=50, seed=0):
    import os, csv, math, copy, random
    os.makedirs("results", exist_ok=True)

    # ---- helpers ----
    def hv_2d(fits, ref):
        if not fits: return 0.0
        pts = sorted(fits, key=lambda x: (x[0], x[1]))  # f1 asc
        hv, prev_f2 = 0.0, ref[1]
        for f1, f2 in pts:
            w  = max(0.0, ref[0] - f1)
            h  = max(0.0, prev_f2 - f2)
            hv += w * h
            prev_f2 = f2
        return hv

    def spread_delta(front_fits):
        n = len(front_fits)
        if n <= 2: return 0.0
        pts = sorted(front_fits, key=lambda x: (x[0], x[1]))
        d = [math.dist(pts[i], pts[i+1]) for i in range(n-1)]
        dbar = sum(d)/len(d)
        d_f, d_l = d[0], d[-1]
        num = d_f + d_l + sum(abs(di - dbar) for di in d)
        den = d_f + d_l + (len(d))*dbar
        return num/den if den > 0 else 0.0

    def save_metrics_row(path, row):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)

    random.seed(seed)
    op_tokens, op_to_candidates = build_operation_index(scenario_path)
    pop = init_population(pop_size, op_tokens, op_to_candidates)
    fits = [evaluate(ind, scenario_path) for ind in pop]

    metrics_path = "results/nsga_metrics.csv"
    # ★ 이전 실행 파일이 남아 있으면 덮어쓰기 위해 삭제(선택)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    hv_hist = []

    # ★ 고정 참조점(ref0)을 '초기세대' 기준으로 설정하고 끝까지 유지
    ref0 = (max(f[0] for f in fits) * 1.1, max(f[1] for f in fits) * 1.1)

    # ---- gen 0 metrics ----
    fronts = nondominated_sort(fits)
    F1 = [fits[i] for i in fronts[0]]
    row0 = {
        "gen": 0,
        "hv": hv_2d(F1, ref=ref0),          # ★ ref0 사용
        "best_f1": min(f[0] for f in fits),
        "best_f2": min(f[1] for f in fits),
        "spread": spread_delta(F1),
        "n_F1": len(F1),
    }
    save_metrics_row(metrics_path, row0)
    hv_hist.append(row0["hv"])

    # ---- main loop ----
    for g in range(1, generations+1):
        fronts = nondominated_sort(fits)
        cd = crowding_distance(fronts[0], fits)

        # parents
        mating_idx = []
        all_idx = list(range(len(pop)))
        while len(mating_idx) < len(pop):
            mating_idx.append(tournament_select(all_idx, fits, fronts, cd))

        # offspring
        offspring = []
        for i in range(0, len(mating_idx), 2):
            p1 = pop[mating_idx[i]]
            p2 = pop[mating_idx[(i+1) % len(mating_idx)]]
            c1, c2 = crossover(copy.deepcopy(p1), copy.deepcopy(p2))
            mutate(c1, op_to_candidates)
            mutate(c2, op_to_candidates)
            offspring.extend([c1, c2])

        off_fits = [evaluate(ind, scenario_path) for ind in offspring]

        # environmental selection
        union, union_fits = pop + offspring, fits + off_fits
        fronts = nondominated_sort(union_fits)
        new_pop, new_fits = [], []
        for F in fronts:
            if len(new_pop) + len(F) <= len(pop):
                new_pop += [union[i] for i in F]
                new_fits += [union_fits[i] for i in F]
            else:
                cd = crowding_distance(F, union_fits)
                rest = len(pop) - len(new_pop)
                F_sorted = sorted(F, key=lambda i: cd.get(i, 0.0), reverse=True)
                new_pop += [union[i] for i in F_sorted[:rest]]
                new_fits += [union_fits[i] for i in F_sorted[:rest]]
                break
        pop, fits = new_pop, new_fits

        # ---- metrics per generation ----
        fronts = nondominated_sort(fits)
        F1 = [fits[i] for i in fronts[0]]
        row = {
            "gen": g,
            "hv": hv_2d(F1, ref=ref0),       # ★ ref0 고정 사용
            "best_f1": min(f[0] for f in fits),
            "best_f2": min(f[1] for f in fits),
            "spread": spread_delta(F1),
            "n_F1": len(F1),
        }
        save_metrics_row(metrics_path, row)
        hv_hist.append(row["hv"])

    # 최종 파레토 프론트 반환
    fronts = nondominated_sort(fits)
    pareto = [(pop[i], fits[i]) for i in fronts[0]]
    return pareto

def nsga2_run_partial(scenario_path: str,
                      n_max: int,
                      pop_size: int = 40,
                      generations: int = 40,
                      seed: int = 0):
    # (a) 부분 대상 샘플링
    sel_ops, sel_cands = sample_ops_for_nsga(scenario_path, n_max, seed)

    # (b) 초기해
    pop = init_population_partial(pop_size, sel_ops, sel_cands)

    # (c) 세대 반복
    fits = [evaluate_partial(repair_partial(ch, sel_cands), scenario_path, sel_ops) for ch in pop]
    metrics = []
    for gen in range(generations):
        # 토너먼트/크로스/변이 (기존 연산자 재사용해도 됨; op_seq·mac_seq 길이가 sel_ops 길이임)
        # --- 예시: 간단 토너먼트+균등교차+언어한 변이 ---
        new_pop = []
        while len(new_pop) < pop_size:
            a, b = random.sample(range(pop_size), 2)
            pa = pop[a]; pb = pop[b]
            # 균등 교차
            cut = random.randrange(1, len(sel_ops)) if len(sel_ops) > 1 else 1
            child1 = Chromosome(pa.op_seq[:cut]+pb.op_seq[cut:],
                                pa.mac_seq[:cut]+pb.mac_seq[cut:],
                                pa.agv_seq[:cut]+pb.agv_seq[cut:])
            child2 = Chromosome(pb.op_seq[:cut]+pa.op_seq[cut:],
                                pb.mac_seq[:cut]+pa.mac_seq[cut:],
                                pb.agv_seq[:cut]+pa.agv_seq[cut:])
            # 간단 변이(스왑/재할당)
            if len(child1.op_seq) > 1 and random.random() < 0.3:
                i, j = random.sample(range(len(child1.op_seq)), 2)
                child1.op_seq[i], child1.op_seq[j] = child1.op_seq[j], child1.op_seq[i]
                child1.mac_seq[i], child1.mac_seq[j] = child1.mac_seq[j], child1.mac_seq[i]
            if random.random() < 0.3:
                k = random.randrange(len(child1.mac_seq))
                cands = sel_cands.get(child1.op_seq[k], [])
                if cands: child1.mac_seq[k] = random.choice(cands)
            # 두 번째 애도 동일 처리
            if len(child2.op_seq) > 1 and random.random() < 0.3:
                i, j = random.sample(range(len(child2.op_seq)), 2)
                child2.op_seq[i], child2.op_seq[j] = child2.op_seq[j], child2.op_seq[i]
                child2.mac_seq[i], child2.mac_seq[j] = child2.mac_seq[j], child2.mac_seq[i]
            if random.random() < 0.3:
                k = random.randrange(len(child2.mac_seq))
                cands = sel_cands.get(child2.op_seq[k], [])
                if cands: child2.mac_seq[k] = random.choice(cands)

            new_pop.extend([child1, child2])

        pop = [repair_partial(ch, sel_cands) for ch in new_pop[:pop_size]]
        fits = [evaluate_partial(ch, scenario_path, sel_ops) for ch in pop]

        # (옵션) 메트릭 로깅 (HV/스프레드 등)
        hv = hv_2d(fits)
        best_f1 = min(f[0] for f in fits)
        best_f2 = min(f[1] for f in fits)
        metrics.append({"gen": gen, "hv": hv, "best_f1": best_f1, "best_f2": best_f2})

    # 파레토 전선에서 하나(예: f1 기준 최저)를 베스트로 리턴
    fronts = nondominated_sort(fits)                       # ← 비지배 정렬
    pareto  = [(pop[i], fits[i]) for i in fronts[0]]       # ← 파레토 프론트 구성
    return pareto, metrics, sel_ops