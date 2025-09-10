from simulator.engine.simulator import EoModel, Event

class SourceStation(EoModel):
    def __init__(self, name="GEN"):
        super().__init__(name)

    def handle_event(self, evt):
        if evt.event_type in ("material_arrival", "part_arrival"):
            part = evt.payload["part"]
            job = part.job
            fetch_ev = Event(
                "agv_fetch_request",
                {"part": part, "job": job, "source_machine": self.name},
                dest_model="AGVController",
            )
            self.schedule(fetch_ev, 0.0)
            return
        return
