from simulator.engine.simulator import EoModel, Event

class Transducer(EoModel):
    def __init__(self):
        super().__init__('transducer')
        self.completed_jobs = []

    def handle_event(self, evt: Event):
        if evt.event_type=='job_completed':
            # Job 완료만 기록하고, 저장은 시뮬레이션 끝에서 한 번만 수행
            part = evt.payload.get('part')
            if part:
                self.completed_jobs.append(part.job.id)
                print(f"[Transducer] Job {part.job.id} 완료 기록")
    
    def finalize(self):
        """시뮬레이션 완료 후 최종 저장"""
        from simulator.result.recorder import Recorder
        Recorder.save()
        print(f"[Transducer] 총 {len(self.completed_jobs)}개 Job 완료: {self.completed_jobs}")
        print(f"[Transducer] trace 파일 저장 완료")