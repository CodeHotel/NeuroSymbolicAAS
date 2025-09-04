from simulator.engine.simulator import EoModel, Event
from simulator.dispatch.dispatch import FIFO
from simulator.result.recorder import Recorder
from simulator.domain.domain import JobStatus
from collections import deque
import random
from enum import Enum, auto

class OperationStatus(Enum):
    QUEUED = auto()
    RUNNING = auto()
    TRANSFER = auto()
    DONE = auto()

class OperationInfo:
    def __init__(self, operation_id, status, location, input_timestamp=None, output_timestamp=None):
        self.operation_id = operation_id
        self.status = status  # OperationStatus Enum
        self.location = location
        self.input_timestamp = input_timestamp
        self.output_timestamp = output_timestamp

    def to_dict(self):
        return {
            'operation_id': self.operation_id,
            'status': self.status.name,  # Enum의 이름으로 저장
            'location': self.location,
            'input_timestamp': self.input_timestamp,
            'output_timestamp': self.output_timestamp
        }

class Machine(EoModel):
    def __init__(self, name, transfer_map, initial, dispatch_rule='fifo', simulator=None):
        super().__init__(name)
        self.status = initial['status']
        self.queue = deque()
        self.running = None
        self.transfer = transfer_map
        self.dispatch = FIFO() if dispatch_rule=='fifo' else FIFO()
        self.next_available_time = 0.0  # 다음 사용 가능 시간
        self.simulator = simulator  # 시뮬레이터 참조
        
        # Job 상태 관리를 위한 큐들 (OperationInfo 대신 Job 상태 직접 관리)
        self.queued_jobs = deque()  # 대기 중인 Job들
        self.running_jobs = deque()  # 실행 중인 Job들
        self.finished_jobs = []  # 완료된 Job들
        
        # 무한 루프 방지를 위한 전송 횟수 추적
        self.transfer_counts = {}  # {job_id: transfer_count}
        self.max_transfers = 1  # 최대 전송 횟수 (1번 전송 후 현재 기계에서 실행)
        
        # 간단한 AGV 로깅 시스템
        self.agv_logs = []  # AGV 활동 로그
        
    def log_agv_activity(self, activity_type, job_id, destination=None, duration=0.0):
        """AGV 활동 로깅"""
        log_entry = {
            'timestamp': EoModel.get_time(),
            'machine': self.name,
            'activity_type': activity_type,  # 'delivery_start', 'delivery_complete', 'return'
            'job_id': job_id,
            'destination': destination,
            'duration': duration
        }
        self.agv_logs.append(log_entry)
        print(f"[AGV {self.name}] {activity_type}: Job {job_id} → {destination} (시간: {duration:.2f}초)")
        
    def save_agv_logs(self, output_dir='results'):
        """AGV 로그를 엑셀 파일로 저장"""
        if not self.agv_logs:
            return None
            
        import pandas as pd
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.agv_logs)
        filename = f'agv_logs_{self.name}.xlsx'
        filepath = os.path.join(output_dir, filename)
        
        df.to_excel(filepath, index=False)
        print(f"[AGV 로그] {filepath} 저장 완료 ({len(self.agv_logs)}개 로그)")
        
        return filepath

    def handle_event(self, evt):
        et = evt.event_type
        if et in ('material_arrival','part_arrival'):
            part = evt.payload['part']
            self._enqueue(part)

        elif et == 'machine_idle_check':
            self._start_if_possible()

        elif et in ('end_operation','operation_complete'):
            op_id = evt.payload.get('operation_id')
            self._finish(op_id)
            
        elif et == 'agv_delivery_complete':
            # AGV 배송 완료 및 복귀 처리
            payload = evt.payload
            self.log_agv_activity('delivery_complete', payload['job_id'], payload['destination'], payload['delivery_time'])
            self.log_agv_activity('return_home', payload['job_id'], self.name, payload['return_time'])

    def _enqueue(self, part):
        self.queue.append(part)
        
        # current_op()이 None인 경우를 처리
        queue_ops = []
        for p in self.queue:
            current_op = p.job.current_op()
            if current_op:
                queue_ops.append(current_op.id)
            else:
                queue_ops.append('DONE')
        
        current_op = part.job.current_op()
        op_id = current_op.id if current_op else 'DONE'
        current_time = EoModel.get_time()
        
        # Job 상태 업데이트
        part.job.set_status(JobStatus.QUEUED)
        part.job.set_location(self.name)
        
        # 대기 중인 Job으로 추가 (중복 방지)
        if part.job not in self.queued_jobs:
            self.queued_jobs.append(part.job)
        
        Recorder.log_queue(part, self.name, current_time, op_id, len(self.queue), queue_ops)

        ev = Event('machine_idle_check', dest_model=self.name)
        self.schedule(ev, 0)

    def _start_if_possible(self):
        if self.status!='idle' or not self.queue:
            return
        
        # release_time이 지난 작업만 처리 가능한지 확인
        current_time = EoModel.get_time()
        available_parts = []
        for part in self.queue:
            if current_time >= part.job.release_time:
                available_parts.append(part)
        
        if not available_parts:
            # 아직 릴리스되지 않은 작업들만 있는 경우 대기
            return
            
        # 동적 스케줄링 사용 여부 확인
        # 정적 스케줄링만 사용하므로 이 부분은 무시됩니다.
        # 대신 큐에서 작업을 선택하여 실행합니다.
        part = self.dispatch.select(available_parts)
        
        if part in self.queue:
            self.queue.remove(part)

        op = part.job.current_op()
        if op is None:
            print(f"[Machine {self.name}] Job {part.job.id}의 현재 Operation이 없음 - 큐에서 제거")
            # 완료된 Job을 큐에서 제거
            if part in self.queue:
                self.queue.remove(part)
            if part.job in self.queued_jobs:
                self.queued_jobs.remove(part.job)
            # 다음 작업 처리
            ev = Event('machine_idle_check', dest_model=self.name)
            self.schedule(ev, 0)
            return
            
        # 정적 스케줄링 모드에서만 assigned 체크
        # 정적 스케줄링이므로 이 부분은 무시됩니다.
        # assigned = op.select_machine()
        # if assigned != self.name:
        #     print(f"[Machine {self.name}] Operation {op.id}이 {assigned}에 할당되어 있음")
        #     # 다른 기계로 전송
        #     self._transfer_to_other_machine(part, assigned)
        #     return
            
        dur = op.sample_duration()

        self.status = 'busy'
        self.running = part
        part.status = 'processing'
        current_time = EoModel.get_time()
        
        # Job 상태 업데이트
        part.job.set_status(JobStatus.RUNNING)
        part.job.set_location(self.name)
        
        # Control Tower에 Job 상태 업데이트
        # 정적 스케줄링이므로 이 부분은 무시됩니다.
        # if self.control_tower:
        #     job_status = {
        #         'status': 'running',
        #         'current_machine': self.name,
        #         'current_operation': op.id,
        #         'start_time': current_time,
        #         'estimated_duration': dur
        #     }
        #     self.control_tower.update_job_status(part.job.id, job_status)
        
        # 대기 중에서 실행 중으로 이동
        if part.job in self.queued_jobs:
            self.queued_jobs.remove(part.job)
        self.running_jobs.append(part.job)
        
        # 기계 상태 업데이트
        
        Recorder.log_start(part, self.name, current_time, op.id, len(self.queue))
        
        # 작업 완료 이벤트 스케줄링
        ev = Event('end_operation', {'part': part, 'operation_id': op.id}, dest_model=self.name)
        self.schedule(ev, dur)

    def _transfer_to_other_machine(self, part, target_machine):
        """다른 기계로 작업 전송"""
        # 현재 큐에서 제거
        if part in self.queue:
            self.queue.remove(part)
        if part.job in self.queued_jobs:
            self.queued_jobs.remove(part.job)
        
        # 전송 시간 계산 (분포에서 샘플링)
        transfer_spec = self.transfer.get(target_machine, {})
        if transfer_spec:
            dist = transfer_spec.get('distribution')
            if dist == 'normal':
                transfer_time = max(0, random.gauss(transfer_spec['mean'], transfer_spec['std']))
            elif dist == 'uniform':
                transfer_time = random.uniform(transfer_spec.get('low', 0), transfer_spec.get('high', 0))
            elif dist == 'exponential':
                transfer_time = random.expovariate(transfer_spec['rate'])
            else:
                transfer_time = 0.0
        else:
            transfer_time = 0.0
        
        # 전송 이벤트 스케줄링
        ev = Event('part_arrival', {'part': part}, dest_model=target_machine)
        self.schedule(ev, transfer_time)
        
        # 전송 횟수 로그
        transfer_count = self.transfer_counts.get(part.job.id, 0)
        print(f"[전송] {self.name} → {target_machine}: {part.job.id} (전송시간: {transfer_time:.2f}, 전송횟수: {transfer_count})")

    def _finish(self, op_id=None):
        part = self.running
        current_time = EoModel.get_time()
        
        # 🚨 Operation 완료 시간 기록 강화
        current_op = part.job.current_op()
        if current_op:
            # 이전 operation의 완료 시간 기록
            current_op.end_time = current_time
            print(f"[{self.name}] Operation {current_op.id} 완료 시간 기록: {current_time:.3f}")
        
        # Job 완료 시간 업데이트
        part.job.set_completion_time(current_time)
        
        # 실행 중에서 제거
        if part.job in self.running_jobs:
            self.running_jobs.remove(part.job)
        
        Recorder.log_end(part, self.name, current_time, op_id)

        part.job.advance()
        if part.job.done():
            # Job 완료 (중복 방지)
            part.job.set_status(JobStatus.DONE)
            part.job.set_location(None)
            if part.job not in self.finished_jobs:
                self.finished_jobs.append(part.job)
            
            # queue와 queued_jobs에서 제거
            if part in self.queue:
                self.queue.remove(part)
            if part.job in self.queued_jobs:
                self.queued_jobs.remove(part.job)
            
            # Job 완료 처리
            
            Recorder.log_done(part, EoModel.get_time())
            done_ev = Event('job_completed', {'part': part}, dest_model='transducer')
            self.schedule(done_ev, 0)
        else:
            if part in self.queue:
                self.queue.remove(part)
            # 다음 기계로 전송
            current_op = part.job.current_op()
            nxt = current_op.select_machine() if current_op else None
            
            # 시뮬레이션 기반 최적화를 위한 기계 선택
            if nxt is None:
                candidates = part.job.current_op().candidates
                if candidates:
                    # 시뮬레이션 기반 최적화에서는 최적화 알고리즘이 결정해야 함
                    # 여기서는 기본 휴리스틱으로 첫 번째 후보 선택 (임시)
                    nxt = candidates[0]
                    print(f"[시뮬레이션 기반 할당] Job {part.job.id}의 {part.job.current_op().id}를 {nxt}로 할당 (최적화 알고리즘이 결정해야 함)")
                else:
                    print(f"경고: Job {part.job.id}의 Operation {part.job.current_op().id}에 후보 기계가 없습니다.")
                    # Job을 완료된 것으로 처리 (중복 방지)
                    part.job.set_status(JobStatus.DONE)
                    part.job.set_location(None)
                    if part.job not in self.finished_jobs:
                        self.finished_jobs.append(part.job)
                    
                    # Job 완료 처리
                    
                    Recorder.log_done(part, EoModel.get_time())
                    done_ev = Event('job_completed', {'part': part}, dest_model='transducer')
                    self.schedule(done_ev, 0)
                    self.running = None
                    self.status = 'idle'
                    ev = Event('machine_idle_check', dest_model=self.name)
                    self.schedule(ev, 0)
                    return
            
            spec = self.transfer.get(nxt, {})
            dist = spec.get('distribution')
            if dist=='normal':
                delay = max(0, random.gauss(spec['mean'], spec['std']))
            elif dist=='uniform':
                delay = random.uniform(spec.get('low',0), spec.get('high',0))
            elif dist=='exponential':
                delay = random.expovariate(spec['rate'])
            else:
                # 🚨 기본 전송 시간 설정으로 겹치는 문제 방지
                delay = 1.0  # 최소 1초 전송 시간 보장
                print(f"[{self.name}] {nxt}로의 전송 시간이 정의되지 않음 - 기본값 {delay}초 사용")

            # Job 상태를 TRANSFER로 설정
            part.job.set_status(JobStatus.TRANSFER)
            part.job.set_location(f"{self.name}->{nxt}")
            
            # queued_jobs에서 제거 (전송 중이므로)
            if part.job in self.queued_jobs:
                self.queued_jobs.remove(part.job)
            
            # 정적 스케줄링이므로 이 부분은 무시됩니다.
            # if self.control_tower:
            #     job_status = {
            #         'status': 'transfer',
            #         'from_machine': self.name,
            #         'to_machine': nxt,
            #         'transfer_time': delay,
            #         'next_operation': part.job.current_op().id
            #     }
            #     self.control_tower.update_job_status(part.job.id, job_status)

            # AGV 배송 시작 로깅
            self.log_agv_activity('delivery_start', part.job.id, nxt, delay)
            
            Recorder.log_transfer(part, self.name, nxt, EoModel.get_time(), delay)

            ev = Event('part_arrival', {'part': part}, dest_model=nxt)
            self.schedule(ev, delay)
            
            # AGV 복귀 시간 계산 (배송 시간과 동일하다고 가정)
            return_delay = delay
            
            # AGV 배송 완료 및 복귀 로깅을 위한 이벤트 스케줄링
            agv_return_ev = Event('agv_delivery_complete', {
                'job_id': part.job.id,
                'destination': nxt,
                'delivery_time': delay,
                'return_time': return_delay
            }, dest_model=self.name)
            self.schedule(agv_return_ev, delay + return_delay)
        
        self.running = None
        self.status = 'idle'
        
        # 기계 상태 업데이트
        
        ev = Event('machine_idle_check', dest_model=self.name)
        self.schedule(ev, 0)

    def get_queue_status(self):
        print(f"\n=== {self.name} 큐 상태 ===")
        print(f"현재 상태: {self.status}")
        print(f"대기 중인 파트 수: {len(self.queue)}")
        
        print(f"\n대기 중인 Job들 (queued_jobs):")
        if self.queued_jobs:
            for i, job in enumerate(self.queued_jobs):
                print(f"  {i+1}. Job {job.id}, Part {job.part_id}, Operation {job.current_op().id if job.current_op() else 'None'}")
                print(f"      상태: {job.status.name}, 위치: {job.current_location}, 진행률: {job.get_progress():.2f}")
        else:
            print("  비어있음")
            
        print(f"\n실행 중인 Job들 (running_jobs):")
        if self.running_jobs:
            for i, job in enumerate(self.running_jobs):
                print(f"  {i+1}. Job {job.id}, Part {job.part_id}, Operation {job.current_op().id if job.current_op() else 'None'}")
                print(f"      상태: {job.status.name}, 위치: {job.current_location}, 진행률: {job.get_progress():.2f}")
        else:
            print("  비어있음")
            
        print(f"\n현재 큐의 operation 목록:")
        if self.queue:
            for i, part in enumerate(self.queue):
                op = part.job.current_op()
                job = part.job
                print(f"  {i+1}. Part {part.id}, Operation {op.id if op else 'None'}")
                print(f"      Job 상태: {job.status.name}, 진행률: {job.get_progress():.2f}")
        else:
            print("  비어있음")
        print("=" * 30)

    def clear_queues(self):
        self.queued_jobs.clear()
        self.running_jobs.clear()
        self.finished_jobs.clear()
        
    def get_job_status_summary(self):
        """현재 머신에서 관리하는 모든 Job의 상태 요약을 반환합니다."""
        summary = {
            'machine_name': self.name,
            'queued_jobs': [job.to_dict() for job in self.queued_jobs],
            'running_jobs': [job.to_dict() for job in self.running_jobs],
            'finished_jobs': [job.to_dict() for job in self.finished_jobs],
            'total_jobs': len(self.queued_jobs) + len(self.running_jobs) + len(self.finished_jobs)
        }
        return summary
