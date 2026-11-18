"""步骤共识:由若干维修单的实际操作顺序求违反票数最少的完整步骤顺序。

模型
----
- 一次登记给出 2~8 个步骤词(完整顺序的取值域),以及至多 30 条现存记录
  (编号 + 该记录实际执行的部分步骤顺序,每条至少两步)。
- 每条有效记录对自身顺序中出现的每一对步骤投一票:靠前的步骤先于靠后的步骤。
  例:记录顺序 [A, C, B] 投出 A<C、A<B、C<B 三票。
- 在步骤词的全部排列中,找违反票数最少者;同票数取字典序最小者。
- 只有仍然有效的记录参与计算。登记时记录下每篇文档的版本号;文档被更正或删除后
  版本号改变(删除后重新录入也会得到新版本号),旧登记中的该条记录永久失效,
  同号重录不会让旧证据复活。

这里的函数只做纯计算/校验,不持有锁;由 ``IndexManager`` 在写锁内调用校验与登记,
并用不可变 ``State`` 快照保证计算只看到同一版语料和同一整组登记。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

MIN_STEPS = 2
MAX_STEPS = 8
MAX_RECORDS = 30


class ConsensusValidationError(Exception):
    """登记参数不合法。整组拒绝,调用方应保留旧登记。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RegistrationEntry:
    """登记中的一条记录:编号、该记录的部分步骤顺序、登记时看到的文档版本。"""

    id: str
    order: tuple[str, ...]
    doc_version: int


@dataclass(frozen=True)
class Registration:
    """一次完整登记(整组)。任何一条记录不合法都不会生成 Registration。"""

    steps: tuple[str, ...]
    entries: tuple[RegistrationEntry, ...]
    corpus_version: int


@dataclass(frozen=True)
class RecordConflicts:
    """某条记录中与最终完整顺序相反的步骤对(按记录中的先后排列)。"""

    id: str
    pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ConsensusResult:
    """共识结果。status 为 ``ok`` 或 ``insufficient_evidence``。"""

    status: str
    order: tuple[str, ...]
    conflicts: int
    records: tuple[RecordConflicts, ...]
    valid_records: int


def _validate_steps(steps: Sequence[str]) -> tuple[str, ...]:
    if not MIN_STEPS <= len(steps) <= MAX_STEPS:
        raise ConsensusValidationError(
            "step_count",
            f"步骤词须为 {MIN_STEPS}~{MAX_STEPS} 个,收到 {len(steps)} 个",
        )
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, str) or not step:
            raise ConsensusValidationError("invalid_step", "步骤词必须是非空字符串")
        if step in seen:
            raise ConsensusValidationError("duplicate_step", f"步骤词重复: {step}")
        seen.add(step)
    return tuple(steps)


def build_registration(
    steps: Iterable[str],
    records: Iterable[tuple[str, Sequence[str]]],
    state: Any,
) -> Registration:
    """校验整组登记并生成不可变 Registration;任一条不合法即抛异常。

    ``records`` 为 [(编号, 该记录的部分步骤顺序)];``state`` 需提供 ``docs``
    (编号 -> (正文, 分词序列)、``doc_versions``(编号 -> 版本号)和 ``version``。
    校验和登记在调用方的同一把写锁内完成,因此记录的版本号与登记内容来自同一版语料。
    """
    step_tuple = _validate_steps(list(steps))
    step_set = set(step_tuple)
    record_list = list(records)
    if len(record_list) > MAX_RECORDS:
        raise ConsensusValidationError(
            "too_many_records", f"记录至多 {MAX_RECORDS} 条,收到 {len(record_list)} 条"
        )

    seen_ids: set[str] = set()
    entries: list[RegistrationEntry] = []
    for doc_id, order in record_list:
        if not isinstance(doc_id, str) or not doc_id:
            raise ConsensusValidationError("invalid_record_id", "记录编号必须是非空字符串")
        if doc_id in seen_ids:
            raise ConsensusValidationError("duplicate_record", f"记录编号重复: {doc_id}")
        seen_ids.add(doc_id)

        doc = state.docs.get(doc_id)
        if doc is None:
            raise ConsensusValidationError("unknown_record", f"记录不存在: {doc_id}")

        order_tuple = tuple(order)
        if len(order_tuple) < 2:
            raise ConsensusValidationError(
                "order_too_short", f"记录 {doc_id} 至少需要两步"
            )
        if len(set(order_tuple)) != len(order_tuple):
            raise ConsensusValidationError(
                "duplicate_order_step", f"记录 {doc_id} 的步骤顺序中有重复步骤"
            )

        tokens = set(doc[1])
        for step in order_tuple:
            if step not in step_set:
                raise ConsensusValidationError(
                    "step_not_registered", f"步骤词 {step} 不在本次登记的步骤词中"
                )
            if step not in tokens:
                raise ConsensusValidationError(
                    "step_not_in_record",
                    f"步骤词 {step} 未出现在记录 {doc_id} 的分词中",
                )

        entries.append(
            RegistrationEntry(
                id=doc_id,
                order=order_tuple,
                doc_version=state.doc_versions.get(doc_id, 0),
            )
        )

    return Registration(
        steps=step_tuple,
        entries=tuple(entries),
        corpus_version=state.version,
    )


def entry_is_valid(entry: RegistrationEntry, state: Any) -> bool:
    """登记时的文档版本与当前状态一致才算有效。

    删除后再以同一编号录入会得到新的版本号,因此旧登记不会因同号重录而恢复。
    """
    return (
        entry.id in state.docs
        and state.doc_versions.get(entry.id) == entry.doc_version
    )


def _pair_counts(entries: Sequence[RegistrationEntry]) -> dict[tuple[str, str], int]:
    """按有向步骤对聚合票数:(先, 后) -> 票数。"""
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        order = entry.order
        for i in range(len(order)):
            earlier = order[i]
            for j in range(i + 1, len(order)):
                key = (earlier, order[j])
                counts[key] = counts.get(key, 0) + 1
    return counts


def _best_order(steps: Sequence[str], pair_counts: dict[tuple[str, str], int]):
    """最少违反票数的完整顺序;同票取字典序最小。

    子集动态规划:把步骤从左到右排入,把 y 接在当前前缀 S 之后时,所有
    "y 先于 S 中某步"的票都会因 y 排在其后而违反,代价为 Σ_{z∈S} w[y][z]。
    h[S] = 排完剩余步骤的最小追加代价。步骤至多 8 个,2^8 个状态即可精确求解。
    重建时每步按步骤词字典序尝试,只接受仍能达到最优代价的候选,从而同票取字典序最小。
    """
    m = len(steps)
    index = {step: i for i, step in enumerate(steps)}
    # w[i][j] = 投票"i 应先于 j"的票数
    w = [[0] * m for _ in range(m)]
    for (earlier, later), count in pair_counts.items():
        w[index[earlier]][index[later]] += count

    full = (1 << m) - 1
    # add_cost[i][mask] = 把 i 接在 mask 之后时的违反票数 Σ_{z∈mask} w[i][z]
    add_cost = [[0] * (full + 1) for _ in range(m)]
    for i in range(m):
        weights = w[i]
        row = add_cost[i]
        for mask in range(1, full + 1):
            low = mask & -mask
            j = low.bit_length() - 1
            row[mask] = row[mask ^ low] + weights[j]

    h = [0] * (full + 1)  # h[full] = 0
    for mask in range(full - 1, -1, -1):
        best = None
        remaining = full ^ mask
        while remaining:
            low = remaining & -remaining
            i = low.bit_length() - 1
            candidate = add_cost[i][mask] + h[mask | low]
            if best is None or candidate < best:
                best = candidate
            remaining ^= low
        h[mask] = best

    by_name = sorted(range(m), key=lambda i: steps[i])
    order: list[str] = []
    mask = 0
    while mask != full:
        for i in by_name:
            bit = 1 << i
            if mask & bit:
                continue
            if add_cost[i][mask] + h[mask | bit] == h[mask]:
                order.append(steps[i])
                mask |= bit
                break
    return tuple(order), h[0]


def compute_consensus(registration: Registration | None, state: Any) -> ConsensusResult:
    """按当前语料快照重算共识。无登记或已无有效记录时报告证据不足。"""
    if registration is None:
        return ConsensusResult("insufficient_evidence", (), 0, (), 0)

    valid = [e for e in registration.entries if entry_is_valid(e, state)]
    if not valid:
        return ConsensusResult("insufficient_evidence", (), 0, (), 0)

    best_order, best_conflicts = _best_order(
        registration.steps, _pair_counts(valid)
    )

    pos_of = {step: i for i, step in enumerate(best_order)}
    records: list[RecordConflicts] = []
    for entry in valid:
        order = entry.order
        pairs = tuple(
            (order[i], order[j])
            for i in range(len(order))
            for j in range(i + 1, len(order))
            if pos_of[order[i]] > pos_of[order[j]]
        )
        records.append(RecordConflicts(entry.id, pairs))

    return ConsensusResult(
        status="ok",
        order=best_order,
        conflicts=best_conflicts,
        records=tuple(records),
        valid_records=len(valid),
    )
