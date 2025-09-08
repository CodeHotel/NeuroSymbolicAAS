# --- simulator/model/agv.py ---
from simulator.engine.simulator import EoModel, Event
from enum import Enum, auto
import math

class AGVStatus(Enum):
    IDLE = auto()          # 유휴 상태
    FETCHING = auto()      # 머신에서 작업 가져오는 중
    LOADING = auto()       # 작업을 AGV에 적재 중
    DELIVERING = auto()    # 목적지로 이동 중
    UNLOADING = auto()     # 목적지에서 작업 하역 중

class AGV(EoModel):
    def __init__(self, agv_id, speed=1.0, capacity=1):
        super().__init__(f"AGV_{agv_id}")
        self.agv_id = agv_id
        self.speed = speed  # 단위: m/s
        self.capacity = capacity  # 동시에 운반할 수 있는 작업 수
        
        # 상태 관리
        self.status = AGVStatus.IDLE
        self.current_location = None  # 현재 위치 (머신 이름)
        self.destination = None       # 목적지 (머신 이름)
        
        # 작업 관리
        self.carried_jobs = []       # 현재 운반 중인 작업들
        self.current_task = None     # 현재 수행 중인 작업
        
        # 이동 관련
        self.departure_time = 0.0    # 출발 시간
        self.arrival_time = 0.0      # 도착 예정 시간
        self.distance = 0.0          # 이동 거리
        
        # 로깅 관련
        self.logger = None  # AGVLogger 인스턴스
        self.task_start_time = None  # 현재 작업 시작 시간
        
    def set_logger(self, logger):
        """로거 설정"""
        self.logger = logger
        
    def _log_event(self, event_type, details):
        """이벤트 로깅"""
        if self.logger:
            self.logger.log_agv_event(f"AGV_{self.agv_id}", event_type, details)
            
    def _log_status_change(self, old_status, new_status):
        """상태 변화 로깅"""
        if self.logger:
            self.logger.log_agv_status_change(
                f"AGV_{self.agv_id}", 
                old_status.name if old_status else "None", 
                new_status.name, 
                self.current_location
            )
            
    def _log_movement(self, from_location, to_location, distance, travel_time):
        """이동 로깅"""
        if self.logger:
            self.logger.log_agv_movement(
                f"AGV_{self.agv_id}",
                from_location,
                to_location,
                distance,
                self.speed,
                travel_time
            )
            
    def _log_task(self, task_type, source_machine, destination_machine, job_id, start_time, end_time):
        """작업 로깅"""
        if self.logger:
            self.logger.log_agv_task(
                f"AGV_{self.agv_id}",
                task_type,
                source_machine,
                destination_machine,
                job_id,
                start_time,
                end_time
            )
        
    def handle_event(self, event):
        """이벤트 처리"""
        if event.event_type == 'agv_fetch_request':
            self._handle_fetch_request(event.payload)
        elif event.event_type == 'agv_delivery_request':
            self._handle_delivery_request(event.payload)
        elif event.event_type == 'agv_fetch_complete':
            self._handle_fetch_complete(event.payload)
        elif event.event_type == 'agv_delivery_complete':
            self._handle_delivery_complete(event.payload)
        elif event.event_type == 'agv_move_complete':
            self._handle_move_complete(event.payload)
    
    def _handle_fetch_request(self, payload):
        """작업 가져오기 요청 처리"""
        if self.status != AGVStatus.IDLE:
            print(f"[{self.name}] 현재 {self.status.name} 상태로 인해 fetch 요청을 거부합니다.")
            return
            
        part = payload.get('part')          # ✅ Part 받기
        job  = part.job if part else payload['job']  # 하위호환
        source_machine = payload['source_machine']
        
        print(f"[{self.name}] {source_machine}에서 Job {job.id} fetch 시작")
        
        # 로깅
        self._log_event('fetch_request', {'source_machine': source_machine, 'job_id': job.id})

        
        # current_task를 먼저 설정
        self.current_task = {
            'type': 'fetch',
            'source_machine': source_machine,
            'part': part,                   # ✅ Part 저장
            'job': job                      # (옵션) 참고용
     }
        
        # 작업 시작 시간 기록
        self.task_start_time = EoModel.get_time()
        
        # AGV를 source machine으로 이동
        self._move_to(source_machine)
    
    def _handle_delivery_request(self, payload):
        """작업 배송 요청 처리"""
        if self.status not in (AGVStatus.IDLE, AGVStatus.LOADING):
            print(f"[{self.name}] 현재 {self.status.name} 상태로 인해 delivery 요청을 거부합니다.")
            return
            
        part = payload['part']               
        job  = part.job
        destination_machine = payload['destination_machine']
        
        print(f"[{self.name}] {destination_machine}로 Job {job.id} delivery 시작")

        self._log_event('delivery_request', {'destination_machine': destination_machine, 'job_id': job.id})

        self.current_task = {
            'type': 'delivery',
            'destination_machine': destination_machine,
            'part': part,                  
            'job': job
        }

        
        # 작업 시작 시간 기록
        self.task_start_time = EoModel.get_time()
        
        # AGV를 destination machine으로 이동
        self._move_to(destination_machine)
    
    def _move_to(self, destination):
        """지정된 목적지로 이동"""
        if self.current_location == destination:
            # 이미 목적지에 있는 경우
            self._arrive_at_destination()
            return
            
        old_status = self.status
        self.destination = destination
        self.status = AGVStatus.DELIVERING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        # 이동 거리 계산 (간단한 유클리드 거리)
        # 실제 구현에서는 좌표 기반 거리 계산 사용
        self.distance = self._calculate_distance(self.current_location, destination)
        
        # 이동 시간 계산
        travel_time = self.distance / self.speed
        
        # 출발 시간과 도착 시간 설정
        current_time = EoModel.get_time()
        self.departure_time = current_time
        self.arrival_time = current_time + travel_time
        
        print(f"[{self.name}] {self.current_location} → {destination} 이동 시작 (거리: {self.distance:.2f}m, 시간: {travel_time:.2f}초)")
        
        # 이동 로깅
        self._log_movement(self.current_location, destination, self.distance, travel_time)
        
        # 이동 완료 이벤트 스케줄링
        ev = Event('agv_move_complete', {
            'agv_id': self.agv_id,
            'destination': destination
        }, dest_model=self.name)
        self.schedule(ev, travel_time)
    
    def _calculate_distance(self, source, destination):
        """두 머신 간의 거리 계산 (간단한 구현)"""
        # 실제 구현에서는 머신 좌표를 사용하여 정확한 거리 계산
        # 현재는 머신 이름을 기반으로 한 간단한 거리 계산
        
        # 머신 이름에서 숫자 추출 (M1, M2, M3 등)
        try:
            source_num = int(source[1:]) if source and source[0] == 'M' else 0
            dest_num = int(destination[1:]) if destination and destination[0] == 'M' else 0
            # 간단한 거리 계산 (머신 간격을 10m로 가정)
            return abs(dest_num - source_num) * 10.0
        except:
            # 기본 거리
            return 20.0
    
    def _arrive_at_destination(self):
        """목적지 도착 처리"""
        self.current_location = self.destination
        self.destination = None
        
        if self.current_task and self.current_task['type'] == 'fetch':
            self._start_fetching()
        elif self.current_task and self.current_task['type'] == 'delivery':
            self._start_unloading()
    
    def _start_fetching(self):
        """작업 가져오기 시작"""
        old_status = self.status
        self.status = AGVStatus.FETCHING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        source_machine = self.current_task['source_machine']
        job = self.current_task['job']
        
        print(f"[{self.name}] {source_machine}에서 Job {job.id} fetch 중...")
        
        # fetch 완료 이벤트 스케줄링 (간단한 로딩 시간)
        ev = Event('agv_fetch_complete', {
            'agv_id': self.agv_id,
            'job': job,
            'source_machine': source_machine
        }, dest_model=self.name)
        self.schedule(ev, 2.0)  # 2초 로딩 시간
    
    def _start_unloading(self):
        """작업 하역 시작"""
        old_status = self.status
        self.status = AGVStatus.UNLOADING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        destination_machine = self.current_task['destination_machine']
        job = self.current_task['job']
        
        print(f"[{self.name}] {destination_machine}에서 Job {job.id} unload 중...")
        
        # unload 완료 이벤트 스케줄링 (간단한 하역 시간)
        ev = Event('agv_delivery_complete', {
            'agv_id': self.agv_id,
            'job': job,
            'destination_machine': destination_machine
        }, dest_model=self.name)
        self.schedule(ev, 2.0)  # 2초 하역 시간
    
    def _handle_fetch_complete(self, payload):
        """작업 가져오기 완료 처리"""
        part = self.current_task.get('part') if self.current_task else None
        job = payload['job']
        source_machine = payload['source_machine']
        
        # 작업을 AGV에 적재
        self.carried_jobs.append(part)
        old_status = self.status
        self.status = AGVStatus.LOADING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        print(f"[{self.name}] Job {job.id} 적재 완료")
        
        # 작업 로깅
        if self.task_start_time is not None:
            self._log_task('fetch', source_machine, None, job.id, self.task_start_time, EoModel.get_time())
        
        # 적재 완료 후 delivery 요청
        self._request_delivery(part)
    
    def _request_delivery(self, part):
        """배송 요청"""
        # 현재 작업의 다음 operation이 할당된 기계 찾기
        job = part.job
        current_op = job.current_op()
        if current_op and current_op.candidates:
            # 첫 번째 후보 기계로 배송 (실제로는 최적화 알고리즘이 결정)
            destination = current_op.candidates[0]
            
            print(f"[{self.name}] Job {job.id}를 {destination}로 배송 요청")
            
            # AGV 컨트롤러에 배송 요청
            ev = Event('agv_delivery_request', {
            'agv_id': self.agv_id,
            'destination_machine': destination,
            'part': part              # ✅ Part 전달
            }, dest_model='AGVController')
            self.schedule(ev, 0)
        else:
            print(f"[{self.name}] Job {job.id}의 다음 작업이 없습니다.")
            self._return_to_idle()
    
    def _handle_delivery_complete(self, payload):
        """배송 완료 처리"""
        part = self.current_task.get('part') if self.current_task else None
        job  = part.job
        destination_machine = payload['destination_machine']
        
        # 작업을 AGV에서 제거
        if part in self.carried_jobs:
            self.carried_jobs.remove(part)
        
        print(f"[{self.name}] Job {job.id}를 {destination_machine}에 배송 완료")
        
        # 작업 로깅
        if self.task_start_time is not None:
            self._log_task('delivery', None, destination_machine, job.id, self.task_start_time, EoModel.get_time())
        
        # 작업을 목적지 기계에 전달
        ev = Event('part_arrival', {
        'part': part,                   # ✅ 여기가 핵심 수정
        'agv_id': self.agv_id
        }, dest_model=destination_machine)
        self.schedule(ev, 0)
        
        # AGV를 유휴 상태로 전환
        self._return_to_idle()
    
    def _handle_move_complete(self, payload):
        """이동 완료 처리"""
        destination = payload['destination']
        print(f"[{self.name}] {destination} 도착 완료")
        self._arrive_at_destination()
    
    def _return_to_idle(self):
        """AGV를 유휴 상태로 전환"""
        old_status = self.status
        self.status = AGVStatus.IDLE
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        self.current_task = None
        self.task_start_time = None
        print(f"[{self.name}] 유휴 상태로 전환")
        
        # AGV 컨트롤러에 유휴 상태 알림
        ev = Event('agv_idle', {
            'agv_id': self.agv_id,
            'location': self.current_location
        }, dest_model='AGVController')
        self.schedule(ev, 0)
    
    def get_status_info(self):
        """AGV 상태 정보 반환"""
        return {
            'agv_id': self.agv_id,
            'status': self.status.name,
            'current_location': self.current_location,
            'destination': self.destination,
            'carried_jobs': [job.id for job in self.carried_jobs],
            'speed': self.speed,
            'capacity': self.capacity
        }
