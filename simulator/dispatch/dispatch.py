# --- simulator/dispatch/dispatch.py ---
from collections import deque

class DispatchStrategy:
    def select(self, queue):
        raise NotImplementedError

class FIFO(DispatchStrategy):
    def select(self, queue):
        if hasattr(queue, 'popleft'):
            # deque인 경우
            return queue.popleft()
        else:
            # list인 경우
            return queue.pop(0)