# simulator/result/recorder.py

import os
import pandas as pd

class Recorder:
    records = []
    enabled = True  # ← 스위치

    @classmethod
    def reset(cls):
        cls.records.clear()

    @classmethod
    def _on(cls):
        return bool(getattr(cls, "enabled", True))

    @classmethod
    def log_queue(cls, part, machine, time, operation_id, queue_length, queue_ops):
        if not cls._on(): return            # ← 필수
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'queued',
            'time': time,
            'queue_length': queue_length,
            'queue_ops': ','.join(queue_ops)
        })

    @classmethod
    def log_start(cls, part, machine, time, operation_id, queue_length):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'start',
            'time': time,
            'queue_length': queue_length,
            'queue_ops': None
        })

    @classmethod
    def log_end(cls, part, machine, time, operation_id):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'end',
            'time': time,
            'queue_length': None,
            'queue_ops': None
        })

    @classmethod
    def log_fetch(cls, part, src_machine, dest_machine, time, delay):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': part.job.current_op().id if part.job.current_op() else None,
            'machine': f"{src_machine}->{dest_machine}",
            'event': 'fetch',
            'time': time,
            'delay': delay,
            'queue_length': None,
            'queue_ops': None
        })

    @classmethod
    def log_delivery(cls, part, src_machine, dest_machine, time, delay):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': part.job.current_op().id if part.job.current_op() else None,
            'machine': f"{src_machine}->{dest_machine}",
            'event': 'delivery',
            'time': time,
            'delay': delay,
            'queue_length': None,
            'queue_ops': None
        })

    @classmethod
    def log_done(cls, part, time):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': None,
            'machine': None,
            'event': 'done',
            'time': time,
            'queue_length': None,
            'queue_ops': None
        })

    @classmethod
    def save(cls):
        os.makedirs('results', exist_ok=True)
        df = pd.DataFrame(cls.records)
        df.to_csv('results/trace.csv', index=False)
        try:
            df.to_excel('results/trace.xlsx', index=False)
        except ImportError:
            pass
