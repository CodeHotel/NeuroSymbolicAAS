# simulator/control/optimizer.py
from simulator.engine.simulator import EoModel
from typing import List, Dict, Tuple
from dataclasses import dataclass
from simulator.result.recorder import Recorder
import os, csv, json, time, random
from simulator.domain.domain import Job, Part

from collections import OrderedDict 

from simulator.NSGA.NSGA import Chromosome, crossover, mutate, repair_partial
from random import shuffle

@dataclass
class EvalResult:
    chrom: Chromosome
    objectives: Tuple[float, ...]  # (예: (makespan, 이동거리, 지연패널티) 등)

class _LRU:
    def __init__(self, maxsize=10000):
        self.maxsize = int(maxsize)
        self._od = OrderedDict()
    def get(self, k):
        if k not in self._od: return None
        v = self._od.pop(k)
        self._od[k] = v
        return v
    def put(self, k, v):
        if k in self._od:
            self._od.pop(k)
        elif len(self._od) >= self.maxsize:
            self._od.popitem(last=False)
        self._od[k] = v
    def clear(self):
        self._od.clear()

def _canonical_key_for_chrom(ch: "Chromosome") -> tuple:
    triples = [(str(o), str(m), str(a)) for o, m, a in zip(ch.op_seq, ch.mac_seq, ch.agv_seq)]
    triples.sort()  # 순서 불변
    base_key = ("len", len(triples), "assign", tuple(triples))
    return base_key

class OptimizationManager(EoModel):
    """
    특정 이벤트에서 호출되어:
    1) 시뮬레이터 스냅샷
    2) 인구(population) 생성 (n_max개만 인코딩)
    3) 각 개체별: restore → 정책주입 → 시뮬레이션 한 번 완료(run/step_events) → objective 계산
    4) NSGA 선택/교차/변이로 다음 세대 생성 (필요시)
    5) 최종 best(혹은 파레토 중 선택)로 policy를 실제 시뮬레이터에 주입
    """
    def __init__(self, name="OptimizationManager",  pop_size=24, generations=12, nmax_list=[5,10,15,20,25,30]):
        super().__init__(name)
        self.pop_size = pop_size
        self.generations = generations
        self.op_ids, self.op_to_cands = [], {}
        self.default_policy = None   # 기본 콜백 백업용
        self.nmax_list = nmax_list

        self._eval_cache = _LRU(maxsize=100000)

    # ---- 유틸: 현재 시점의 최적화 대상(op, candidates) 수집 ---- 수정 필요 
    def _collect_frontier_ops(self, sim):
        now = sim.now()
        op_ids: list[str] = []
        op_to_cands: dict[str, list[str]] = {}

        processing_jobs = []
        delivery_job = []
        
        ctr = sim.models.get('AGVController')
        src = sim.models.get('GEN')
        
        non_allocation_job = (list(getattr(ctr, 'request_queue', [])))
        for m in sim.machines:
            delivery_job += (list(getattr(m, 'waiting_for_pickup', [])))
            processing_jobs += (list(getattr(m, 'running_jobs', [])))
            processing_jobs += (list(getattr(m, 'queued_jobs', [])))
        delivery_job += (list(getattr(src, 'waiting_for_pickup', [])))

        for a in sim.agvs:
            processing_jobs += (list(getattr(a, 'carried_jobs', [])))

        for part in delivery_job :
            if type(part) is not Part :
                part = Part(part, part.id)

            start = 0
            if part in non_allocation_job:
                start = getattr(part.job, 'idx', -10)
            else :
                start = getattr(part.job, 'idx', -10) + 1
            if(start < 0) :
                raise Exception("Error: job idx is not properly set.")
            ops = getattr(part.job, 'ops', []) or []
            idx = getattr(part.job, 'idx', None)
            
            for fo in ops[start:]:
                cands = getattr(fo, 'candidates', None)
                if not cands:
                    continue  # 후보 없으면 NSGA 인코딩 불가하므로 스킵
                oid = fo.id
                if oid not in op_to_cands:
                    op_to_cands[oid] = list(cands)
                    op_ids.append(oid)

        for part in processing_jobs :
            if type(part) is not Part :
                part = Part(part.id, part)

            start = getattr(part.job, 'idx', -10) + 1
            if(start < 0) :
                raise Exception("Error: job idx is not properly set.")
            ops = getattr(part.job, 'ops', []) or []
            idx = getattr(part.job, 'idx', None)
            
            for fo in ops[start:]:
                cands = getattr(fo, 'candidates', None)
                if not cands:
                    continue  # 후보 없으면 NSGA 인코딩 불가하므로 스킵
                oid = fo.id
                if oid not in op_to_cands:
                    op_to_cands[oid] = list(cands)
                    op_ids.append(oid)

        self.op_ids = op_ids
        self.op_to_cands = op_to_cands
        return op_ids, op_to_cands



    def _plan_from_chrom(self, ch: Chromosome):
        return {op: {"machine": str(mac), "agv": str(agv)}
                for op, mac, agv in zip(ch.op_seq, ch.mac_seq, ch.agv_seq)}

    def _evaluate(self, sim, chrom: Chromosome, fallback_rule) -> EvalResult:
        print(f"[OptimizationManager] 개체 평가: {chrom}")
        ck = _canonical_key_for_chrom(chrom)
        cached = self._eval_cache.get(ck)
        if cached is not None:
            # 동일 (op→machine,agv) 할당이면 시뮬 없이 즉시 반환
            return cached
        snap = sim.snapshot()
        try:
            sim.restore(snap)
            sim._suspend_optimize = True
            sim.optim_plan = self._plan_from_chrom(chrom)   # ← 계획 주입
            print("----------------------------------test------------------------------------")
            sim.run()                                       # 동일 메커니즘으로 평가
            # 다목적: (makespan, agv_total_travel)
            makespan, agv_travel = sim.objective_multi()    # 아래 4)에서 추가
            res = EvalResult(chrom, (makespan, agv_travel))
            self._eval_cache.put(ck, res)
            return res
        finally:
            sim._suspend_optimize = False
            sim.restore(snap)
            sim.optim_plan = {}  # 오염 방지

    def handle_event(self, evt):
        if evt.event_type != 'optimize_now': return
        if getattr(self.simulator, "_suspend_optimize", False): return
        print("[OptimizationManager] 최적화 시작")

        sim = self.simulator
        all_ops, op2cands = self._collect_frontier_ops(sim)

        self._eval_cache.clear()  # 시점 변경 시 캐시 초기화

        agv_ctrl = sim.models.get('AGVController')
        agv_ids = list(getattr(agv_ctrl, 'agvs', {}).keys()) if agv_ctrl else []

        # release_time(=현재 시점)별 상위 폴더
        rt = sim.now()
        rt_str = f"{rt:.3f}".replace('.', '_')   # 예: 123.456 → "123_456"
        base_dir = os.path.join("results", f"rt_{rt_str}")

        os.makedirs(base_dir, exist_ok=True)  # ← 먼저 폴더 생성

        manifest = {
            "release_time": rt,
            "n_max_list": list(self.nmax_list),
            "total_ops": len(all_ops)
        }
        with open(os.path.join(base_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        summary_path = os.path.join(base_dir, "summary.csv")

        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["release_time", "total_ops", "n_max", "makespan", "agv_travel"])

        sim.running_time += time.time() - sim.start_time
        sim.start_time = time.time()
        snap = sim.snapshot()
        for n_max in self.nmax_list:
            self._eval_cache.clear()  # n_max마다 캐시 초기화
            if n_max > len(all_ops):
                continue
            # ===== 1) 진화/평가 단계: 로깅 비활성 =====
            sim.restore(snap)
            start_time = time.time()
            sim.optim_plan = {}
            Recorder.enabled = False  # 평가 중 trace 방지

            pop = self._init_population(all_ops, op2cands, agv_ids, n_max)
            cur_pop, last_scores = pop, None
            for _ in range(self.generations):
                scores = [self._evaluate(sim, ch, None) for ch in cur_pop]
                last_scores = scores
                cur_pop = self._next_generation(cur_pop, scores, op2cands)

            fronts = self._fast_nondominated_sort(last_scores)
            if not (fronts and fronts[0]):
                continue

            knee = self._pick_knee(last_scores, fronts[0])
            best_plan = self._plan_from_chrom(knee.chrom)
            Recorder.set_meta(release_time=rt, n_max=n_max, total_ops=len(all_ops))
            
            opt_time = time.time() - start_time
            sim.running_time += opt_time

            # ===== 2) 로깅 전용 패스: 동일 스냅샷에서 1회 실행 =====
            sim.restore(snap)
            sim.optim_plan = best_plan
            Recorder.reset()
            Recorder.enabled = True

            outdir = os.path.join(base_dir, f"n{n_max}")
            Recorder.set_dir(outdir)   # ← 여기서 폴더 지정 (trace.csv 저장될 곳)


            # 평가 때와 동일 조건으로 ‘한 번’ 돌려서 trace 기록
            sim._suspend_optimize = True   # 평가용 런 중 최적화 재진입 방지
            sim.run()
            sim._suspend_optimize = False

            # 목적값 다시 뽑아 저장(실제 로깅 패스 기준 수치)
            makespan, agv_travel = sim.objective_multi()

            # 파일로 저장
            Recorder.save()  # outdir/trace.csv, trace.xlsx

            # 보조 메타 저장(원하면 해 구조까지 json으로)
            with open(os.path.join(outdir, "objectives.json"), "w") as f:
                json.dump({"n_max": n_max, "makespan": makespan, "agv_travel": agv_travel, "sim_running_time" : sim.running_time}, f, indent=2)
            with open(os.path.join(outdir, "plan.json"), "w") as f:
                json.dump(best_plan, f, indent=2)

            with open(summary_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([rt, len(all_ops), n_max, makespan, agv_travel])

            # 실행적용은 하지 않음(룰 기반 유지)
            sim.optim_plan = {}
            sim.restore(snap)
            Recorder.enabled = False

            # ---- 간단한 NSGA 외피 ----
    def _init_population(self, op_ids, op_to_cands, agv_ids: List[str], n_max) -> List[Chromosome]:
        pop = []
        tmp = op_ids[:]
        for _ in range(self.pop_size):
            shuffle(tmp)
            mac_seq = [random.choice(op_to_cands[op]) for op in tmp]
            agv_seq = [random.choice(agv_ids) if agv_ids else "1" for _ in tmp]
            pop.append(Chromosome(tmp[:n_max], mac_seq[:n_max], agv_seq[:n_max]))
        return pop

    def _next_generation(self, pop: List[Chromosome], scores: List[EvalResult], op_to_cands) -> List[Chromosome]:
        # 1) Fronts 계산
        fronts = self._fast_nondominated_sort(scores)
        # 2) 새 부모 집합 구성 (front 순서대로, crowding 큰 순)
        new_parents: List[Chromosome] = []
        for F in fronts:
            if len(new_parents) + len(F) <= self.pop_size:
                new_parents.extend([scores[i].chrom for i in F])
            else:
                cd = self._crowding_distance(F, scores)
                F_sorted = sorted(F, key=lambda i: cd[i], reverse=True)
                need = self.pop_size - len(new_parents)
                new_parents.extend([scores[i].chrom for i in F_sorted[:need]])
                break

        # 3) 자식 생성
        children: List[Chromosome] = []
        while len(new_parents) + len(children) < self.pop_size:
            p1, p2 = random.sample(new_parents, 2)
            c1, c2 = crossover(p1, p2)
            mutate(c1, op_to_cands, p=0.2)
            mutate(c2, op_to_cands, p=0.2)
            c1 = repair_partial(c1, op_to_cands, self.op_ids)
            c2 = repair_partial(c2, op_to_cands, self.op_ids)
            children.extend([c1, c2])

        return new_parents + children[: self.pop_size - len(new_parents)]

    def _dominates(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        # 모두 최소화 기준
        not_worse = all(x <= y for x, y in zip(a, b))
        strictly_better = any(x < y for x, y in zip(a, b))
        return not_worse and strictly_better

    def _fast_nondominated_sort(self, evals: List[EvalResult]):
        S = {i: [] for i in range(len(evals))}
        n = [0] * len(evals)
        fronts: List[List[int]] = [[]]

        for i, ei in enumerate(evals):
            for j, ej in enumerate(evals):
                if i == j: 
                    continue
                if self._dominates(ei.objectives, ej.objectives):
                    S[i].append(j)
                elif self._dominates(ej.objectives, ei.objectives):
                    n[i] += 1
            if n[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        fronts.pop()  # 마지막 빈 front 제거
        return fronts  # index 리스트의 리스트

    def _crowding_distance(self, indices: List[int], evals: List[EvalResult]) -> Dict[int, float]:
        if not indices:
            return {}
        m = len(evals[0].objectives)
        dist = {i: 0.0 for i in indices}
        # 목적별 정렬 후 경계 무한대, 내부는 보간 거리
        for k in range(m):
            idx_sorted = sorted(indices, key=lambda i: evals[i].objectives[k])
            f = [evals[i].objectives[k] for i in idx_sorted]
            fmin, fmax = f[0], f[-1]
            if fmax == fmin:  # 모두 같은 값이면 거리 기여 0
                continue
            dist[idx_sorted[0]] = float('inf')
            dist[idx_sorted[-1]] = float('inf')
            for r in range(1, len(idx_sorted)-1):
                prev_f = evals[idx_sorted[r-1]].objectives[k]
                next_f = evals[idx_sorted[r+1]].objectives[k]
                dist[idx_sorted[r]] += (next_f - prev_f) / (fmax - fmin + 1e-12)
        return dist
    
    def _normalize_objs(self, pts: List[Tuple[float, ...]]):
        # 최소화 기준, 목적 2개 가정
        import math
        cols = list(zip(*pts))
        mins = [min(c) for c in cols]
        maxs = [max(c) for c in cols]
        def norm(p):
            return tuple((p[i]-mins[i]) / (maxs[i]-mins[i] + 1e-12) for i in range(len(p)))
        return [norm(p) for p in pts]

    def _pick_knee(self, evals: List[EvalResult], front_indices: List[int]) -> EvalResult:
        # a=(0,0), b=(1,1) 직선에서 수직거리 최대
        import math
        P = [evals[i] for i in front_indices]
        N = self._normalize_objs([p.objectives for p in P])
        ax, ay, bx, by = 0.0, 0.0, 1.0, 1.0
        vx, vy = bx-ax, by-ay
        denom = math.hypot(vx, vy) + 1e-12
        def perp_dist(x, y):
            # | (b-a) x (p-a) | / |b-a|
            return abs(vx*(ay - y) - vy*(ax - x)) / denom
        dists = [perp_dist(x, y) for (x, y) in N]
        return P[max(range(len(P)), key=lambda i: dists[i])]
