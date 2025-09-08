from simulator.engine.simulator import EoModel, Event

class AGVController(EoModel):
    def __init__(self, agv_list):
        super().__init__('AGVController')
        self.agvs = {agv.agv_id: agv for agv in agv_list}
        self.idle_pool = set(self.agvs.keys())

    def handle_event(self, evt):
        et = evt.event_type
        p  = evt.payload

        if et == 'agv_fetch_request':
            part = p.get('part')
            job  = part.job if part else p['job']
            src  = p['source_machine']
            agv_id = self._acquire_idle_agv()
            if agv_id is None:
                self.schedule(Event('agv_fetch_request', p, 'AGVController'), 1.0)
                return
            ev = Event('agv_fetch_request',
                       {'part': part, 'job': job, 'source_machine': src},
                       dest_model=f"AGV_{agv_id}")
            self.schedule(ev, 0)

        elif et == 'agv_delivery_request':
            agv_id = p['agv_id']
            dst    = p['destination_machine']
            part   = p['part']
            ev = Event('agv_delivery_request',
                       {'destination_machine': dst, 'part': part},
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
