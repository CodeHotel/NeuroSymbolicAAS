# --- simulator/model/generator.py ---
from simulator.engine.simulator import EoModel, Event
from simulator.domain.domain import Part

class Generator(EoModel):
    def __init__(self, releases, jobs, optimize_on_release=True, optimizer_model='OptimizationManager', epsilon=1e-6):
        super().__init__('generator')
        self.releases, self.jobs = releases, jobs
        self.optimize_on_release = optimize_on_release
        self.optimizer_model = optimizer_model
        self.epsilon = float(epsilon)

        self._jobs_by_id = {jid: job for jid, job in jobs.items()}
        self._parts_by_id = {}  # initialize에서 채움
        
    def initialize(self):
        releases_by_time = {}
        for r in self.releases:
            t = float(r['release_time'])
            releases_by_time.setdefault(t, []).append(r)

        for t, rels in sorted(releases_by_time.items()):
            for r in rels:
                job = self.jobs[r['job_id']]
                part = Part(job.part_id, job)

                # ✅ 생성 즉시 레지스트리에 등록
                self._jobs_by_id[job.id] = job
                self._parts_by_id[part.id] = part

                print(f"[Generator] Job {job.id} (Part {part.id}) 릴리스 시간: {t}")

                candidates = job.current_op().candidates
                if candidates:
                    dest = "GEN"
                    ev = Event('material_arrival', {'part': part}, dest_model=dest)
                    self.schedule(ev, t)
                    print(f"[Generator] Job {job.id}을 {t}초에 {dest}로 전송 예약")
                else:
                    print(f"경고: Job {job.id}의 Operation {job.current_op().id}에 후보 기계가 없습니다.")

            if self.optimize_on_release and self.optimizer_model:
                opt_ev = Event('optimize_now', dest_model=self.optimizer_model)
                self.schedule(opt_ev, t + self.epsilon)

    def handle_event(self, evt): pass

    def save_state(self) -> dict:
        # part→job 링크 테이블과 객체의 경량 상태를 저장
        parts = {}
        for pid, part in (self._parts_by_id or {}).items():
            parts[pid] = part.save_state()  # {'part_id', 'job_id', 'status'}
        jobs = {}
        for jid, job in (self._jobs_by_id or {}).items():
            jobs[jid] = job.save_state()
        return {
            'parts': parts,
            'jobs': jobs,
        }

    def load_state(self, st: dict, reg: dict):
        # 우선 reg 기반으로 존재 객체를 채움(등록 순서 보장)
        self._jobs_by_id = dict(self._jobs_by_id or {})
        self._parts_by_id = dict(self._parts_by_id or {})

        # Job들 복원(이미 존재하면 상태만 갱신)
        for jid, jst in (st.get('jobs', {}) or {}).items():
            job = self._jobs_by_id.get(jid)
            if job is None:
                # 요청대로 자동 생성하지 않음
                raise ValueError(f"Generator.load_state: job {jid} not found (no auto-fix).")
            job.load_state(jst)

        # Part들 복원 및 링크 고정
        for pid, pst in (st.get('parts', {}) or {}).items():
            part = self._parts_by_id.get(pid)
            if part is None:
                raise ValueError(f"Generator.load_state: part {pid} not found (no auto-fix).")
            part.load_state(pst, reg)