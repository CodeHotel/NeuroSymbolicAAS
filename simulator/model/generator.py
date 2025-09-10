# --- simulator/model/generator.py ---
from simulator.engine.simulator import EoModel, Event
from simulator.domain.domain import Part

class Generator(EoModel):
    def __init__(self, releases, jobs):
        super().__init__('generator')
        self.releases, self.jobs = releases, jobs

    def initialize(self):
        for r in self.releases:
            job = self.jobs[r['job_id']]
            part = Part(job.part_id, job)  # jobs에서 part_id를 가져옴
            release_time = r['release_time']
            
            print(f"[Generator] Job {job.id} (Part {part.id}) 릴리스 시간: {release_time}")
            
            # 동적 스케줄링에서는 초기 할당을 하지 않고, 
            # 모든 가능한 기계에 job을 대기 상태로 배치
            candidates = job.current_op().candidates
            if candidates:
                # 첫 번째 후보 기계에만 전송 (실제 할당은 최적화 알고리즘이 결정)
                dest = "GEN"
                ev = Event('material_arrival', {'part': part}, dest_model=dest)
                self.schedule(ev, release_time)
                print(f"[Generator] Job {job.id}을 {release_time}초에 {dest}로 전송 예약 (동적 할당 대기)")
            else:
                print(f"경고: Job {job.id}의 Operation {job.current_op().id}에 후보 기계가 없습니다.")

    def handle_event(self, evt): pass