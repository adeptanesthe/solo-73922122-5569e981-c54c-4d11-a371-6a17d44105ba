"""步骤共识:纯计算、校验、记录失效与 API 行为。"""
import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app.consensus import (
    ConsensusValidationError,
    build_registration,
    compute_consensus,
)
from app.index import IndexManager, build_state
from app.main import create_app


def state_with(docs):
    """docs: {id: tokens};正文无关紧要。"""
    return build_state({i: (i, tuple(t)) for i, t in docs.items()})


# ---------- 纯计算 ----------


def test_single_record_order_has_zero_conflicts():
    st = state_with({"R1": ["A", "B", "C"]})
    reg = build_registration(["A", "B", "C"], [("R1", ["C", "A", "B"])], st)
    result = compute_consensus(reg, st)
    assert result.status == "ok"
    assert result.order == ("C", "A", "B")
    assert result.conflicts == 0
    assert result.valid_records == 1
    assert result.records[0].pairs == ()


def test_majority_wins_and_conflicting_pairs_reported():
    st = state_with({"R1": ["A", "B", "C"], "R2": ["A", "B", "C"], "R3": ["A", "B", "C"]})
    reg = build_registration(
        ["A", "B", "C"],
        [("R1", ["B", "A"]), ("R2", ["A", "B"]), ("R3", ["A", "B", "C"])],
        st,
    )
    result = compute_consensus(reg, st)
    assert result.order == ("A", "B", "C")
    assert result.conflicts == 1  # 仅 R1 的 B<A 被违反
    by_id = {r.id: r.pairs for r in result.records}
    assert by_id["R1"] == (("B", "A"),)
    assert by_id["R2"] == () and by_id["R3"] == ()


def test_partial_order_pairs_only_among_present_steps():
    # R1 顺序 [C, A]:只投 C<A 一票,不涉及 B
    st = state_with({"R1": ["A", "B", "C"]})
    reg = build_registration(["A", "B", "C"], [("R1", ["C", "A"])], st)
    result = compute_consensus(reg, st)
    assert result.conflicts == 0
    # 满足 C<A 且 0 冲突的排列有 (B,C,A)、(C,A,B)、(C,B,A);取字典序最小的 (B,C,A)
    assert result.order == ("B", "C", "A")


def test_tie_break_lexicographically_smallest():
    # A<B 与 B<A 各一票:票价相同,取字典序最小的 (A,B)
    st = state_with({"R1": ["A", "B"], "R2": ["A", "B"]})
    reg = build_registration(["A", "B"], [("R1", ["A", "B"]), ("R2", ["B", "A"])], st)
    result = compute_consensus(reg, st)
    assert result.order == ("A", "B")
    assert result.conflicts == 1


def brute_force(steps, entries):
    """独立参照:枚举全部排列,按记录逐对计票。"""
    best = None
    for order in itertools.permutations(steps):
        pos = {s: i for i, s in enumerate(order)}
        conflicts = 0
        for entry in entries:
            o = entry.order
            for i in range(len(o)):
                for j in range(i + 1, len(o)):
                    if pos[o[i]] > pos[o[j]]:
                        conflicts += 1
        key = (conflicts, order)
        if best is None or key < best:
            best = key
    return best


def test_dp_matches_brute_force_random():
    rng = random.Random(11)
    for trial in range(120):
        m = rng.randint(2, 6)
        steps = [f"S{i}" for i in range(m)]
        docs = {}
        entries_req = []
        for r in range(rng.randint(1, 8)):
            rid = f"R{r}"
            k = rng.randint(2, m)
            order = rng.sample(steps, k)
            docs[rid] = steps  # 每条记录分词包含全部步骤
            entries_req.append((rid, order))
        st = state_with(docs)
        reg = build_registration(steps, entries_req, st)
        result = compute_consensus(reg, st)
        expected = brute_force(steps, reg.entries)
        assert (result.conflicts, result.order) == expected


def test_all_eight_steps_solvable():
    steps = [f"P{i}" for i in range(8)]
    st = state_with({"R1": steps})
    reg = build_registration(steps, [("R1", steps[::-1])], st)
    result = compute_consensus(reg, st)
    assert result.conflicts == 0
    assert result.order == tuple(reversed(steps))


def test_no_registration_is_insufficient():
    st = state_with({"R1": ["A", "B"]})
    result = compute_consensus(None, st)
    assert result.status == "insufficient_evidence"
    assert result.order == () and result.records == () and result.valid_records == 0


# ---------- 校验:整组拒绝 ----------


def test_unknown_record_rejected():
    st = state_with({"R1": ["A", "B"]})
    with pytest.raises(ConsensusValidationError) as ei:
        build_registration(["A", "B"], [("R1", ["A", "B"]), ("NOPE", ["A", "B"])], st)
    assert ei.value.code == "unknown_record"


def test_duplicate_steps_rejected():
    st = state_with({"R1": ["A", "B"]})
    with pytest.raises(ConsensusValidationError) as ei:
        build_registration(["A", "A"], [("R1", ["A", "B"])], st)
    assert ei.value.code == "duplicate_step"


def test_duplicate_step_in_record_order_rejected():
    st = state_with({"R1": ["A", "B"]})
    with pytest.raises(ConsensusValidationError) as ei:
        build_registration(["A", "B"], [("R1", ["A", "A"])], st)
    assert ei.value.code == "duplicate_order_step"


def test_step_not_in_record_tokens_rejected():
    st = state_with({"R1": ["A", "B"]})
    with pytest.raises(ConsensusValidationError) as ei:
        build_registration(["A", "B", "C"], [("R1", ["A", "C"])], st)
    assert ei.value.code == "step_not_in_record"


def test_order_shorter_than_two_rejected():
    st = state_with({"R1": ["A", "B"]})
    with pytest.raises(ConsensusValidationError):
        build_registration(["A", "B"], [("R1", ["A"])], st)


def test_step_count_and_record_count_limits():
    st = state_with({"R1": ["A"]})
    with pytest.raises(ConsensusValidationError):
        build_registration(["A"], [], st)
    with pytest.raises(ConsensusValidationError):
        build_registration([f"S{i}" for i in range(9)], [], st)
    docs = {f"R{i}": ["A", "B"] for i in range(31)}
    st2 = state_with(docs)
    with pytest.raises(ConsensusValidationError):
        build_registration(["A", "B"], [(f"R{i}", ["A", "B"]) for i in range(31)], st2)


# ---------- API ----------


@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def put_docs(client, docs):
    r = client.post(
        "/documents/batch",
        json={
            "upserts": [
                {"id": i, "text": i, "tokens": toks} for i, toks in docs.items()
            ],
            "deletes": [],
        },
    )
    assert r.status_code == 200
    return r


def register(client, steps, records):
    return client.post(
        "/consensus/register",
        json={"steps": steps, "records": [{"id": i, "order": o} for i, o in records]},
    )


def test_api_register_and_query(client):
    put_docs(client, {"R1": ["A", "B", "C"], "R2": ["A", "B", "C"]})
    r = register(client, ["A", "B", "C"], [("R1", ["B", "A", "C"]), ("R2", ["A", "B"])])
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "ok",
        "order": ["A", "B", "C"],
        "conflicts": 1,
        "records": [
            {"id": "R1", "pairs": [["B", "A"]]},
            {"id": "R2", "pairs": []},
        ],
        "valid_records": 2,
    }
    assert client.get("/consensus").json() == body


def test_api_failed_registration_keeps_old_group(client):
    put_docs(client, {"R1": ["A", "B", "C"]})
    assert register(client, ["A", "B", "C"], [("R1", ["C", "A"])]).status_code == 200
    before = client.get("/consensus").json()
    bad = register(client, ["A", "B", "C"], [("R1", ["C", "A"]), ("MISSING", ["A", "B"])])
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "unknown_record"
    assert client.get("/consensus").json() == before  # 旧组仍可用


def test_correction_invalidates_old_order_and_recomputes(client):
    put_docs(client, {"R1": ["A", "B", "C"], "R2": ["A", "B", "C"]})
    register(client, ["A", "B", "C"], [("R1", ["C", "A"]), ("R2", ["A", "B"])])
    assert client.get("/consensus").json()["valid_records"] == 2
    # 更正 R1 后旧证据失效,只剩 R2
    put_docs(client, {"R1": ["A", "B", "C"]})
    body = client.get("/consensus").json()
    assert body["valid_records"] == 1
    assert body["order"] == ["A", "B", "C"]
    assert [r["id"] for r in body["records"]] == ["R2"]


def test_delete_then_readd_same_id_does_not_restore_old_evidence(client):
    put_docs(client, {"R1": ["A", "B", "C"]})
    register(client, ["A", "B", "C"], [("R1", ["C", "A"])])
    client.post("/documents/batch", json={"upserts": [], "deletes": ["R1"]})
    assert client.get("/consensus").json()["status"] == "insufficient_evidence"
    put_docs(client, {"R1": ["A", "B", "C"]})
    body = client.get("/consensus").json()
    assert body["status"] == "insufficient_evidence"  # 同号重录不恢复旧顺序
    assert body["valid_records"] == 0


def test_all_records_invalid_reports_insufficient(client):
    put_docs(client, {"R1": ["A", "B"]})
    register(client, ["A", "B"], [("R1", ["B", "A"])])
    put_docs(client, {"R1": ["A", "B"]})
    body = client.get("/consensus").json()
    assert body == {
        "status": "insufficient_evidence",
        "order": [],
        "conflicts": 0,
        "records": [],
        "valid_records": 0,
    }


def test_failed_correction_preserves_previous_conclusion(client):
    from app.index import MAX_DOCS

    put_docs(client, {"R1": ["A", "B"]})
    register(client, ["A", "B"], [("R1", ["B", "A"])])
    before = client.get("/consensus").json()
    big = [{"id": f"X{i}", "text": "t", "tokens": ["A"]} for i in range(MAX_DOCS)]
    r = client.post("/documents/batch", json={"upserts": big, "deletes": []})
    assert r.status_code == 409
    assert client.get("/consensus").json() == before


def test_api_validation_returns_422_and_legacy_search_still_works(client):
    put_docs(client, {"R1": ["A", "B"]})
    for payload in (
        {"steps": ["A"], "records": []},
        {"steps": ["A", "A"], "records": [{"id": "R1", "order": ["A", "B"]}]},
        {"steps": ["A", "B"], "records": [{"id": "R1", "order": ["A"]}]},
        {"steps": ["A", "B"], "records": [{"id": "R1", "order": ["A", "A"]}]},
        {"steps": ["A", "B", "C"], "records": [{"id": "R1", "order": ["A", "C"]}]},
        {"steps": ["A", "B"], "records": [{"id": "NOPE", "order": ["A", "B"]}]},
        {"steps": ["A", "B"], "records": [{"id": "R1", "order": ["A", "B"]}, {"id": "R1", "order": ["A", "B"]}]},
    ):
        assert client.post("/consensus/register", json=payload).status_code == 422
    # 检索不受影响
    assert client.post("/search", json={"terms": ["A"], "k": 5}).json()["hits"][0]["id"] == "R1"
    assert client.post("/search/exhaustive", json={"terms": ["A"], "k": 5}).status_code == 200


# ---------- 交错与同一版本 ----------


def test_registration_observes_corrections_between_calls(client):
    """登记与更正交错:每次查询只用同一版语料 + 相容的整组顺序。"""
    put_docs(client, {"R1": ["A", "B", "C"], "R2": ["A", "B", "C"]})
    register(client, ["A", "B", "C"], [("R1", ["C", "B", "A"]), ("R2", ["C", "B"])])
    first = client.get("/consensus").json()
    assert first["order"] == ["C", "B", "A"] and first["valid_records"] == 2

    put_docs(client, {"R2": ["A", "B", "C"]})  # 更正 R2
    second = client.get("/consensus").json()
    assert second["valid_records"] == 1
    assert [r["id"] for r in second["records"]] == ["R1"]
    assert second["order"] == ["C", "B", "A"]


def test_concurrent_register_and_correction_never_mixes_versions(client):
    """并发登记/更正:结果要么对应登记时快照,要么按更新语料重算,不出现撕裂。"""
    import threading

    put_docs(client, {f"R{i}": ["A", "B", "C"] for i in range(10)})
    register(client, ["A", "B", "C"], [(f"R{i}", ["C", "B", "A"]) for i in range(10)])
    errors = []

    def writer():
        try:
            for _ in range(10):
                put_docs(client, {"R0": ["A", "B", "C"]})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(10):
                r = client.get("/consensus")
                assert r.status_code == 200
                body = r.json()
                assert body["status"] in ("ok", "insufficient_evidence")
                # 登记时的 R0 可能已被更正,故为 9 或 10;绝不会是撕裂的中间值
                assert body["valid_records"] in (0, 9, 10)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    # 最终 R0 已更正,只剩 9 条有效
    final = client.get("/consensus").json()
    assert final["valid_records"] == 9


def test_post_consensus_alias_and_docs_entry(client):
    put_docs(client, {"R1": ["A", "B"]})
    r = client.post(
        "/consensus",
        json={"steps": ["A", "B"], "records": [{"id": "R1", "order": ["B", "A"]}]},
    )
    assert r.status_code == 200 and r.json()["order"] == ["B", "A"]
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
