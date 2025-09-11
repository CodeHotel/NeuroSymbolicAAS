from dataclasses import dataclass
from typing import List
import random

@dataclass
class Chromosome:
    op_seq: List[str]   # 전역 공정 순서 (operation_id 리스트)
    mac_seq: List[str]  # 각 op에 할당할 머신 이름
    agv_seq: List[str]  # 각 op에 할당할 AGV id (현재는 placeholder)

def crossover(p1: Chromosome, p2: Chromosome):
    # op_seq: OX(순서 교차) 대용 - 구간을 보존하고 나머지는 상대부모 순서로 채움
    def ox(a, b):
        L = len(a); i, j = sorted(random.sample(range(L), 2))
        mid  = a[i:j]
        tail = [x for x in b if x not in mid]
        return tail[:i] + mid + tail[i:]

    c1_seq = ox(p1.op_seq, p2.op_seq)
    c2_seq = ox(p2.op_seq, p1.op_seq)

    # mac_seq / agv_seq: 균등 교차(uniform)
    def uni(a, b): 
        return [random.choice([x, y]) for x, y in zip(a, b)]

    c1 = Chromosome(c1_seq, uni(p1.mac_seq, p2.mac_seq), uni(p1.agv_seq, p2.agv_seq))
    c2 = Chromosome(c2_seq, uni(p1.mac_seq, p2.mac_seq), uni(p1.agv_seq, p2.agv_seq))
    return c1, c2

def mutate(ind: Chromosome, op_to_candidates, p=0.1):
    # op_seq: 확률 p로 두 위치 스왑
    if random.random() < p:
        i, j = sorted(random.sample(range(len(ind.op_seq)), 2))
        ind.op_seq[i], ind.op_seq[j] = ind.op_seq[j], ind.op_seq[i]

    # mac_seq: 각 위치를 확률 p로 해당 op의 후보 머신 중 무작위 재할당
    for t, oid in enumerate(ind.op_seq):
        if random.random() < p:
            cands = op_to_candidates.get(oid, [])
            if cands:
                ind.mac_seq[t] = random.choice(cands)

def repair_partial(ch: Chromosome, op_to_candidates):
    fixed = False
    mac_seq = ch.mac_seq[:]
    for i, oid in enumerate(ch.op_seq):
        cands = op_to_candidates.get(oid, [])
        if not cands:
            continue
        if mac_seq[i] not in cands:
            mac_seq[i] = random.choice(cands)  # 후보군에 맞게 치환
            fixed = True
    return Chromosome(ch.op_seq[:], mac_seq, ch.agv_seq[:]) if fixed else ch
