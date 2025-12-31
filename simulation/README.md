# AASX (Advanced Adaptive Scheduling eXperiment)

시뮬레이터 기반 최적화를 통한 제조 시스템 스케줄링 최적화 도구입니다.

## 주요 기능

- **시뮬레이터 기반 최적화**: 시뮬레이터를 사용하여 가능한 모든 스케줄링 조합을 탐색하고 최적해를 찾습니다
- **다양한 최적화 알고리즘**: DFS, Branch-and-Bound, MCTS 알고리즘 지원
- **정확한 시간 축 관리**: 이벤트 기반 시뮬레이션으로 정확한 시간 계산
- **상세한 결과 분석**: 최적 스케줄과 예측 완료 시간 제공

## 설치 및 실행

### 1. 시나리오 파일 준비

`scenarios/my_case/` 디렉토리에 다음 파일들을 준비하세요:

```
scenarios/my_case/
├── jobs.json              # 작업 정의
├── operations.json        # 작업 단계 정의
├── machines.json          # 기계 정의
├── operation_durations.json  # 작업 시간 분포
├── machine_transfer_time.json  # 기계 간 이동 시간
├── initial_machine_status.json # 초기 기계 상태
└── job_release.json       # 작업 출시 시간
```

### 2. 프로그램 실행

```bash
# 기본 실행 (Branch-and-Bound 알고리즘)
python simulator/main.py --scenario scenarios/my_case

# 다른 알고리즘 사용
python simulator/main.py --algorithm mcts --scenario scenarios/my_case

# 시간 제한 설정
python simulator/main.py --time_limit 600 --scenario scenarios/my_case

# 최대 노드 수 설정
python simulator/main.py --max_nodes 50000 --scenario scenarios/my_case
```

### 3. 명령행 옵션

- `--algorithm`: 최적화 알고리즘 선택 (`dfs`, `branch_and_bound`, `mcts`)
- `--policy`: 롤아웃 정책 선택 (`ect`, `spt`, `atc`, `edd`)
- `--time_limit`: 최적화 시간 제한 (초)
- `--max_nodes`: 최대 탐색 노드 수
- `--scenario`: 시나리오 디렉토리 경로

## 출력 결과

프로그램 실행 후 `results/` 디렉토리에 다음 파일들이 생성됩니다:

- `simulator_optimization_result.json`: 최적화 결과 (최적 스케줄, makespan 등)
- `trace.xlsx`: 상세 스케줄링 결과 (Excel 형식)
- `job_info.csv`: 작업별 완료 시간 정보
- `operation_info.csv`: 작업 단계별 정보

## 최적화 과정

1. **시나리오 로드**: JSON 파일들에서 시뮬레이션 모델 생성
2. **최적화 실행**: 시뮬레이터를 사용하여 가능한 모든 스케줄링 조합 탐색
3. **최적해 선택**: 가장 좋은 makespan을 가진 스케줄 선택
4. **검증 시뮬레이션**: 찾은 최적 스케줄로 시뮬레이션 실행
5. **결과 생성**: 상세한 스케줄링 결과와 분석 데이터 생성

## 알고리즘 설명

### Branch-and-Bound
- 가장 정확한 최적화 알고리즘
- 가지치기를 통한 효율적인 탐색
- 전역 최적해 보장

### MCTS (Monte Carlo Tree Search)
- 확률적 탐색 알고리즘
- 큰 문제에서도 효율적
- 근사 최적해 제공

### DFS (Depth-First Search)
- 완전 탐색 알고리즘
- 작은 문제에서 사용
- 모든 가능한 조합 탐색

## 예시 결과

```json
{
  "algorithm": "BRANCH_AND_BOUND",
  "best_objective": 15.2,
  "search_time": 45.3,
  "nodes_explored": 5000,
  "best_schedule": [
    "Action(O11 -> M1, pos=None)",
    "Action(O12 -> M2, pos=None)",
    "Action(O21 -> M1, pos=None)"
  ]
}
```

## 라이선스

MIT License