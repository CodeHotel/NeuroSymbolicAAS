# --- simulator/model/machine_agv.py ---
from simulator.engine.simulator import EoModel, Event
from enum import Enum, auto

class MachineAGVStatus(Enum):
    IDLE = auto()          # 유휴 상태 (머신에 대기)
    DELIVERING = auto()    # 작업 배송 중
    RETURNING = auto()     # 배송 완료 후 돌아오는 중

class MachineAGV(EoModel):
    def __init__(self, machine_name, speed=1.0):
        super().__init__(f"AGV_{machine_name}")
        self.machine_name = machine_name  # 소속 머신
        self.speed = speed  # 단위: m/s
        
        # 상태 관리
        self.status = MachineAGVStatus.IDLE
        self.current_location = machine_name  # 현재 위치 (항상 소속 머신)
        self.destination = None       # 목적지 (배송할 머신)
        
        # 작업 관리
        self.carried_job = None      # 현재 운반 중인 작업
        self.delivery_task = None    # 배송 작업 정보
        
        # 이동 관련
        self.departure_time = 0.0    # 출발 시간
        self.arrival_time = 0.0      # 도착 예정 시간
        self.distance = 0.0          # 이동 거리
        
        # 로깅 관련
        self.logger = None
        self.task_start_time = None
        
    def set_logger(self, logger):
        """로거 설정"""
        self.logger = logger
        
    def _log_event(self, event_type, details):
        """이벤트 로깅"""
        if self.logger:
            self.logger.log_agv_event(f"AGV_{self.machine_name}", event_type, details)
            
    def _log_status_change(self, old_status, new_status):
        """상태 변화 로깅"""
        if self.logger:
            self.logger.log_agv_status_change(
                f"AGV_{self.machine_name}", 
                old_status.name if old_status else "None", 
                new_status.name, 
                self.current_location
            )
            
    def _log_movement(self, from_location, to_location, distance, travel_time):
        """이동 로깅"""
        if self.logger:
            self.logger.log_agv_movement(
                f"AGV_{self.machine_name}",
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
                f"AGV_{self.machine_name}",
                task_type,
                source_machine,
                destination_machine,
                job_id,
                start_time,
                end_time
            )
        
    def deliver_job(self, job, destination_machine):
        """작업을 목적지 머신으로 배송"""
        if self.status != MachineAGVStatus.IDLE:
            print(f"[{self.name}] 현재 {self.status.name} 상태로 인해 배송 요청을 거부합니다.")
            return
            
        print(f"[{self.name}] {destination_machine}로 Job {job.id} 배송 시작")
        
        # 로깅
        self._log_event('delivery_request', {
            'destination_machine': destination_machine,
            'job_id': job.id
        })
        
        # 배송 작업 정보 설정
        self.delivery_task = {
            'job': job,
            'destination': destination_machine,
            'source': self.machine_name
        }
        
        # 작업 시작 시간 기록
        self.task_start_time = EoModel.get_time()
        
        # 목적지로 이동
        self._move_to(destination_machine)
        
    def _move_to(self, destination):
        """지정된 목적지로 이동"""
        if self.current_location == destination:
            # 이미 목적지에 있는 경우
            self._arrive_at_destination()
            return
            
        old_status = self.status
        self.destination = destination
        self.status = MachineAGVStatus.DELIVERING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        # 이동 거리 계산
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
            'destination': destination
        }, dest_model=self.name)
        self.schedule(ev, travel_time)
        
    def _calculate_distance(self, source, destination):
        """두 머신 간의 거리 계산"""
        try:
            source_num = int(source[1:]) if source and source[0] == 'M' else 0
            dest_num = int(destination[1:]) if destination and destination[0] == 'M' else 0
            # 머신 간격을 10m로 가정
            return abs(dest_num - source_num) * 10.0
        except:
            return 20.0
            
    def _arrive_at_destination(self):
        """목적지 도착 처리"""
        if self.status == MachineAGVStatus.DELIVERING:
            self._deliver_job()
        elif self.status == MachineAGVStatus.RETURNING:
            self._return_to_home()
            
    def _deliver_job(self):
        """작업 배송 처리"""
        if not self.delivery_task:
            return
            
        job = self.delivery_task['job']
        destination = self.delivery_task['destination']
        
        print(f"[{self.name}] {destination}에 Job {job.id} 배송 완료")
        
        # 작업을 목적지 기계에 전달
        ev = Event('part_arrival', {
            'part': job,
            'agv_id': self.machine_name
        }, dest_model=destination)
        self.schedule(ev, 0)
        
        # 작업 로깅
        if self.task_start_time is not None:
            self._log_task('delivery', self.machine_name, destination, job.id, self.task_start_time, EoModel.get_time())
        
        # 작업 정보 정리
        self.carried_job = None
        self.delivery_task = None
        
        # 소속 머신으로 돌아가기
        self._return_home()
        
    def _return_home(self):
        """소속 머신으로 돌아가기"""
        if self.current_location == self.machine_name:
            # 이미 집에 있는 경우
            self._return_to_home()
            return
            
        old_status = self.status
        self.status = MachineAGVStatus.RETURNING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        # 집으로 이동
        self._move_to(self.machine_name)
        
    def _return_to_home(self):
        """집에 도착하여 유휴 상태로 전환"""
        self.current_location = self.machine_name
        self.destination = None
        
        old_status = self.status
        self.status = MachineAGVStatus.IDLE
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        print(f"[{self.name}] {self.machine_name}에 도착하여 유휴 상태로 전환")
        
    def handle_event(self, event):
        """이벤트 처리"""
        if event.event_type == 'agv_move_complete':
            self._handle_move_complete(event.payload)
            
    def _handle_move_complete(self, payload):
        """이동 완료 처리"""
        destination = payload['destination']
        print(f"[{self.name}] {destination} 도착 완료")
        self._arrive_at_destination()
        
    def get_status_info(self):
        """AGV 상태 정보 반환"""
        return {
            'agv_id': f"AGV_{self.machine_name}",
            'machine_name': self.machine_name,
            'status': self.status.name,
            'current_location': self.current_location,
            'destination': self.destination,
            'carried_job': self.carried_job.id if self.carried_job else None,
            'speed': self.speed
        }

