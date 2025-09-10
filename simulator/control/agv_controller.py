from simulator.engine.simulator import EoModel, Event

class AGVController(EoModel):
    def __init__(self, agv_list):
        super().__init__('AGVController')
        self.agvs = {agv.agv_id: agv for agv in agv_list}
        self.idle_pool = set(self.agvs.keys())

    def save_state(self):
        # 현재 보유 AGV ID들과 idle 풀만 저장 (객체는 그대로 유지)
        return {
            "agv_ids": list(self.agvs.keys()),
            "idle_pool": list(self.idle_pool),
        }

    def load_state(self, st, registry=None):
        # 기존 self.agvs(객체 매핑)는 그대로 두고, idle_pool만 복원
        saved_ids = st.get("agv_ids")
        if saved_ids:
            # 혹시라도 agvs가 비었다면, saved_ids를 바탕으로 안전 복구
            if not getattr(self, "agvs", None):
                self.agvs = {agv_id: self.agvs.get(agv_id) for agv_id in saved_ids if self.agvs and agv_id in self.agvs}
            # registry로부터 보강(선택)
            if registry and hasattr(registry, "get"):
                # 필요하면 registry에서 AGV 객체 재결합하는 로직을 여기에 추가
                pass
        # idle 풀 복원 (없으면 모든 AGV를 idle로 가정)
        self.idle_pool = set(st.get("idle_pool", saved_ids or list(self.agvs.keys())))

    def _earliest_available_time(self):
        now = EoModel.get_time()
        # idle이면 지금(now), 아니면 그 AGV의 arrival_time(없으면 무한대)
        etas = []
        for agv in self.agvs.values():
            if agv.agv_id in self.idle_pool:
                etas.append(now)
            else:
                etas.append(getattr(agv, 'arrival_time', float('inf')))
        return (min(etas) if etas else now)

    # ====== ETA 계산 유틸 ======
    def _lookup_transfer_time(self, agv, source: str, destination: str):
        """AGV 내부의 전이시간 맵/헬퍼가 있으면 그것을 우선 사용."""
        # 1) 사내 헬퍼 메서드 우선
        if hasattr(agv, '_lookup_transfer_time'):
            try:
                return agv._lookup_transfer_time(source, destination)
            except Exception:
                pass
        # 2) transfer_time_map 직접 참조
        tt = getattr(agv, 'transfer_time_map', None)
        if isinstance(tt, dict):
            if (source, destination) in tt:
                return tt[(source, destination)]
        # 3) 동일 위치면 0, 아니면 None
        if source == destination:
            return 0.0
        return None

    def _eta_from_agv_to_src(self, agv, src: str):
        """
        해당 agv가 지금 src로 가기까지의 ETA(초)를 추정.
        - idle 상태 가정 (busy면 여기서는 선발대상 아님)
        - 거리/시간 정보 없으면 큰 값으로 처리
        """
        # 현재 위치
        cur = getattr(agv, 'current_location', None)
        if cur is None:
            # 위치 정보가 없다면 보수적으로 큰 값
            return float('inf')
        # 전이시간 조회
        base_time = self._lookup_transfer_time(agv, cur, src)
        if base_time is None:
            # 맵에 없으면 보수적으로 큰 값
            return float('inf')
        speed = max(1e-9, float(getattr(agv, 'speed', 1.0)))  # 0 나눗셈 방지
        return float(base_time) / speed

    def handle_event(self, evt):
        et = evt.event_type
        p  = evt.payload

        if et == 'agv_fetch_request':
            part = p.get('part')
            job  = part.job if part else p['job']
            src  = p['source_machine']
            # ▶ ETA 최소 AGV 선택
            agv_id = self._acquire_idle_agv(src)
            if agv_id is None:
                t = self._earliest_available_time()
                delay = max(0, t - EoModel.get_time()) + 0.1
                self.schedule(Event('agv_fetch_request', p, 'AGVController'), delay)
                return
            ev = Event('agv_fetch_request',
                       {'part': part,
                        'job': job,
                        'source_machine': src},
                       dest_model=f"AGV_{agv_id}")
            self.schedule(ev, 0)

        elif et == 'agv_delivery_request':
            agv_id = p['agv_id']
            part   = p['part']
            ev = Event('agv_delivery_request',
                       {'part' : part},
                       dest_model=f"AGV_{agv_id}")
            self.schedule(ev, 0)

        elif et == 'agv_idle':
            # AGV가 유휴 전환을 알림
            self.idle_pool.add(p['agv_id'])

    def _acquire_idle_agv(self, src: str = None):
        """idle 풀에서 하나 선택. src가 있으면 ETA 최소 AGV를 고른다."""
        if not self.idle_pool:
            return None
        if src is None:
            # 호환성: 기존처럼 하나 꺼내기
            agv_id = next(iter(self.idle_pool))
            self.idle_pool.remove(agv_id)
            return agv_id

        # ETA 최소화 선택
        best_id = None
        best_eta = float('inf')
        for agv_id in list(self.idle_pool):
            agv = self.agvs.get(agv_id)
            eta = self._eta_from_agv_to_src(agv, src)
            if eta < best_eta:
                best_eta = eta
                best_id = agv_id

        if best_id is None:
            # 모든 ETA를 계산할 수 없을 때, 폴백: 하나 꺼냄
            best_id = next(iter(self.idle_pool))
        self.idle_pool.remove(best_id)
        return best_id
