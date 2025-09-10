from simulator.engine.simulator import EoModel, Event

class AGVController(EoModel):
    def __init__(self, agv_list):
        super().__init__('AGVController')
        self.agvs = {agv.agv_id: agv for agv in agv_list}
        self.idle_pool = set(self.agvs.keys())

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

    def handle_event(self, evt):
        et = evt.event_type
        p  = evt.payload

        if et == 'agv_fetch_request':
            part = p.get('part')
            job  = part.job if part else p['job']
            src  = p['source_machine']
            agv_id = self._acquire_idle_agv()
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

    def _acquire_idle_agv(self):
        if not self.idle_pool: return None
        agv_id = next(iter(self.idle_pool))
        self.idle_pool.remove(agv_id)
        return agv_id
