from simulator.engine.simulator import Simulator
from simulator.builder import ModelBuilder
import pandas as pd
import os
import argparse
from simulator.NSGA.NSGA import nsga2_run
from simulator.NSGA.pin_api import PinSpec, decode_pins
from typing import List

def _op_mean(op) -> float:
    d = getattr(op, 'distribution', {}) or {}
    if 'mean' in d: return float(d['mean'])                               # e.g. normal
    if 'low' in d and 'high' in d: return (float(d['low'])+float(d['high']))/2.0  # uniform
    if 'rate' in d and float(d['rate'])>0: return 1.0/float(d['rate'])    # exponential
    # duration 필드가 있다면 fallback
    dur = getattr(op, 'duration', None)
    try:
        return float(dur) if dur is not None else 0.0
    except:
        return 0.0

def _est_input_load(machine_name: str) -> float:
    m = m_by_name.get(machine_name)
    if m is None: return float('inf')
    total = 0.0
    # 그 머신 입력큐의 작업들의 현재 op 평균 처리시간 합
    for part in getattr(m, 'queue', []):
        op = part.job.current_op()
        if op: total += _op_mean(op)
    return total

def rule_next_machine(job_id: str, op_id: str, candidates: List[str]):
    if not candidates: return None
    # 입력큐 처리시간 합이 최소인 머신 선택 (동률시 min 사전순)
    return min(candidates, key=_est_input_load)

def print_all_machine_queues(machines):
    """모든 기계의 큐 상태를 출력"""
    print("\n" + "="*50)
    print("모든 기계의 큐 상태")
    print("="*50)
    for machine in machines:
        # 각 머신이 get_queue_status()를 제공한다고 가정
        if hasattr(machine, "get_queue_status"):
            machine.get_queue_status()
        else:
            # 대체 출력
            qsize = len(getattr(machine, "queue", []))
            print(f"- {getattr(machine, 'name', 'machine')} | queue size = {qsize}")

def save_all_job_info(machines, filename='results/job_info.csv'):
    """모든 Job의 정보를 CSV 파일로 저장"""
    all_jobs = []
    for machine in machines:
        # 머신이 queued_jobs/running_jobs/finished_jobs 속성을 가진다고 가정
        for attr, qtype in [('queued_jobs','queued'), ('running_jobs','running'), ('finished_jobs','finished')]:
            jobs = getattr(machine, attr, [])
            for job in jobs:
                # job이 to_dict()를 제공한다고 가정
                jd = job.to_dict() if hasattr(job, "to_dict") else {}
                jd['job_id'] = getattr(job, 'id', None)
                jd['machine'] = getattr(machine, 'name', 'unknown')
                jd['queue_type'] = qtype
                all_jobs.append(jd)

    os.makedirs('results', exist_ok=True)
    if all_jobs:
        df = pd.DataFrame(all_jobs)
        df.to_csv(filename, index=False)
        print(f"[저장 완료] {filename} (rows={len(all_jobs)})")
    else:
        # 빈 파일이라도 내려주기
        pd.DataFrame(all_jobs).to_csv(filename, index=False)
        print(f"[경고] 저장할 Job 정보가 없습니다. (빈 CSV 저장) -> {filename}")

def save_all_operation_info(machines, filename='results/operation_info.csv'):
    """기존 호환성을 위한 함수 - Job 정보를 Operation 형태로 변환하여 저장 (가능한 정보만)"""
    all_ops = []
    for machine in machines:
        # 실행 중인 Job들에서 현재 Operation 정보 추출
        for job in getattr(machine, 'running_jobs', []):
            cur_op = job.current_op() if hasattr(job, "current_op") else None
            if cur_op:
                op_dict = {
                    'operation_id': getattr(cur_op, 'id', None),
                    'job_id': getattr(job, 'id', None),
                    'status': getattr(job, 'status', getattr(job, 'state', 'running')) if hasattr(job, 'status') else 'running',
                    'location': getattr(job, 'current_location', getattr(machine, 'name', 'unknown')),
                    'input_timestamp': getattr(job, 'last_completion_time', None),
                    'output_timestamp': None
                }
                all_ops.append(op_dict)

        # 완료된 Job들에서 Operation 정보 추출 (유추 기반)
        for job in getattr(machine, 'finished_jobs', []):
            ops = getattr(job, 'ops', [])
            for i, op in enumerate(ops):
                op_dict = {
                    'operation_id': getattr(op, 'id', None),
                    'job_id': getattr(job, 'id', None),
                    'status': 'completed',
                    'location': getattr(machine, 'name', f"Machine_{i+1}"),
                    'input_timestamp': getattr(job, 'release_time', None),
                    'output_timestamp': getattr(job, 'completion_time', None)
                }
                all_ops.append(op_dict)

    os.makedirs('results', exist_ok=True)
    if all_ops:
        df = pd.DataFrame(all_ops)
        df.to_csv(filename, index=False)
        print(f"[저장 완료] {filename} (rows={len(all_ops)})")
    else:
        pd.DataFrame(all_ops).to_csv(filename, index=False)
        print(f"[경고] 저장할 operation 정보가 없습니다. (빈 CSV 저장) -> {filename}")

def replay_solution(ch, scenario_path: str, results_dir: str = "results"):
    from simulator.result.recorder import Recorder
    Recorder.enabled = True    # 최종 실행에서는 기록 켬
    Recorder.records.clear()   # 이전 평가에서 쌓인 로그 완전히 비움
    """최종 염색체로 한 번 더 시뮬레이터를 돌려서 trace/CSV/엑셀 로그 생성."""
    from simulator.result.recorder import Recorder
    if hasattr(Recorder, "reset"):
        Recorder.reset()
    elif hasattr(Recorder, "clear"):
        Recorder.clear()
    import os
    os.makedirs(results_dir, exist_ok=True)

    # 모델 구성
    builder = ModelBuilder(scenario_path, use_dynamic_scheduling=True)
    machines, gen, tx = builder.build()

    # 시뮬레이터 생성/등록
    sim = Simulator()
    for m in machines:
        m.simulator = sim
        sim.register(m)
    sim.register(gen)
    sim.register(tx)

    # 선택 콜백 주입 (op_id -> machine)
    plan = {op_id: mac for op_id, mac in zip(ch.op_seq, ch.mac_seq)}
    def select_next_machine(job_id: str, op_id: str, candidates):
        sel = plan.get(op_id)
        return sel if (candidates and sel in candidates) else (candidates[0] if candidates else None)
    sim.select_next_machine = select_next_machine

    # 초기화 및 실행
    if hasattr(gen, "initialize"):
        gen.initialize()
    sim.run()

    # 트랜스듀서 최종 저장 (trace.csv/.xlsx 등)
    if hasattr(tx, "finalize"):
        tx.finalize()

    # (선택) 머신별 AGV 로그 저장
    for m in machines:
        if hasattr(m, "save_agv_logs"):
            try:
                m.save_agv_logs(results_dir)
            except Exception as e:
                print(f"[경고] {getattr(m,'name','machine')} AGV 로그 저장 실패: {e}")


if __name__ == '__main__':
    # 명령행 인수 파싱 (최적화 관련 옵션 제거)
    parser = argparse.ArgumentParser(description='시뮬레이터 실행기 (NSGA-II 지원)')
    parser.add_argument('--scenario', default='scenarios/my_case', help='시나리오 경로')
    parser.add_argument('--print_queues_interval', type=float, default=None, help='큐 상태 출력 주기(초)')
    parser.add_argument('--print_job_summary_interval', type=float, default=None, help='Job 요약 출력 주기(초)')

    # ▼ NSGA 관련
    parser.add_argument('--use_nsga', action='store_true', help='NSGA-II로 최적화 실행')
    parser.add_argument('--pop', type=int, default=50, help='NSGA 인구수')
    parser.add_argument('--gen', type=int, default=50, help='NSGA 세대수')
    parser.add_argument('--seed', type=int, default=0, help='NSGA 시드')

    # 룰 기반 관련
    # 기존 인자들 아래에 추가
    parser.add_argument('--pins', type=str, default=None,
                    help='핀셋 매핑 JSON 경로: [{"job_id":"J1","op_id":"O11","machine":"M1","agv":"AGV_1"}, ...]')
    # RL 대비해서 n_max도 받고만 있기 - 나중에 삭제 필요 
    parser.add_argument('--n_max', type=int, default=None, help='(미사용) 추후 RL용 슬롯 크기')

    
    args = parser.parse_args()
    scenario_path = args.scenario

    if args.use_nsga:
        print("="*60)
        print("NSGA-II 최적화 실행")
        print("="*60)
        pareto = nsga2_run(scenario_path, pop_size=args.pop, generations=args.gen, seed=args.seed)
        print("\n[파레토 프론트 상위] (makespan, total_agv_time)")
        for i, (ch, (f1, f2)) in enumerate(pareto[:10], 1):
            print(f"{i:2d}) Cmax={f1:.3f}, AGVmove={f2:.3f}")
        best_ch, (best_cmax, best_agv) = pareto[0]
        print(f"\n[Best 선택] Cmax={best_cmax:.3f}, AGVmove={best_agv:.3f} → 로그 생성")
        replay_solution(best_ch, scenario_path, results_dir='results')
        raise SystemExit(0)

    print("="*60)
    print("단일 패스 시뮬레이션 실행")
    print("="*60)
    print(f"시나리오: {scenario_path}")
    print("="*60)

    # 시뮬레이터 설정
    # 기존과 동일하게 use_dynamic_scheduling=True 유지 (동적 스케줄링 사용 시)
    builder = ModelBuilder(scenario_path, use_dynamic_scheduling=True)
    machines, gen, tx = builder.build()

    # Simulator 생성
    sim = Simulator()

    # 모델 등록
    # (엔진이 각 모델의 simulator 참조가 필요하면 설정)
    for m in machines:
        m.simulator = sim
        sim.register(m)
    sim.register(gen)
    sim.register(tx)

   # 핀셋 로드 (없으면 빈 dict)
    sim.pins = {}
    if args.pins:
        import json
        with open(args.pins, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pin_list = [PinSpec(**item) for item in data]  # item: {job_id, op_id, machine?, agv?}
        sim.pins = decode_pins(pin_list)

    m_by_name = {m.name: m for m in machines}

    # Generator 초기화 (이벤트 초기 생성)
    if hasattr(gen, "initialize"):
        gen.initialize()

    # 시뮬레이션 실행 (한 번만)
    sim.run(print_queues_interval=args.print_queues_interval, 
            print_job_summary_interval=args.print_job_summary_interval)

    # 시뮬레이션 완료 후 상태 출력
    print("\n시뮬레이션 완료 후 상태:")
    print_all_machine_queues(machines)

    # 최종 Job 상태 요약 출력
    print("\n최종 Job 상태 요약:")
    if hasattr(sim, "print_job_status_summary"):
        sim.print_job_status_summary()

    # transducer finalize 호출하여 trace.xlsx/trace.csv 생성 (구현에 따라)
    if hasattr(tx, "finalize"):
        tx.finalize()
        print("\nresults/trace.xlsx (또는 trace.csv)가 생성되었는지 확인하세요.")

    # 결과 파일명 설정
    job_info_file = 'results/job_info.csv'
    operation_info_file = 'results/operation_info.csv'

    # 모든 Job 정보 저장
    save_all_job_info(machines, job_info_file)

    # 기존 호환성을 위한 Operation 정보 저장
    save_all_operation_info(machines, operation_info_file)

    # AGV 로그 저장 (옵션)
    print(f"\n=== AGV 로그 저장 시도 ===")
    agv_files_saved = []
    for machine in machines:
        if hasattr(machine, 'save_agv_logs'):
            try:
                agv_log_file = machine.save_agv_logs('results')
                if agv_log_file:
                    agv_files_saved.append(agv_log_file)
            except Exception as e:
                print(f"- {getattr(machine,'name','machine')} AGV 로그 저장 실패: {e}")

    if agv_files_saved:
        print(f"=== AGV 로그 파일 생성 완료 ===")
        for file_path in agv_files_saved:
            print(f"- {file_path}")
    else:
        print("AGV 로그가 없거나 저장에 실패했습니다.")

    print("\n" + "="*60)
    print("시뮬레이션 완료! (최적화 없음)")
    print("="*60)
    print("결과 파일:")
    print(f"- {job_info_file}: Job 정보")
    print(f"- {operation_info_file}: Operation 정보")
    print(f"- results/trace.csv / results/trace.xlsx: 시뮬레이션 이벤트 로그")