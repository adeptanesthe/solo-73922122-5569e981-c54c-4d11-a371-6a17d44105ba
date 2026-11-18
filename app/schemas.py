"""请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


class ConsensusRecord(BaseModel):
    """登记中的一条记录:编号与该记录实际执行的部分步骤顺序。"""

    id: str = Field(min_length=1, max_length=128, description="现存维修记录编号")
    order: list[str] = Field(
        min_length=2, description="该记录实际执行的步骤顺序,至少两步"
    )


class ConsensusRegisterRequest(BaseModel):
    """整组登记:2~8 个步骤词 + 至多 30 条现存记录。"""

    steps: list[str] = Field(
        min_length=2, max_length=8, description="步骤词(顺序取值域),不得重复"
    )
    records: list[ConsensusRecord] = Field(
        max_length=30, description="现存记录编号与各自部分步骤顺序,至多 30 条"
    )


class RecordConflicts(BaseModel):
    id: str
    pairs: list[list[str]] = Field(
        description="该记录中与最终完整顺序相反的步骤对,按记录内先后排列"
    )


class ConsensusResponse(BaseModel):
    status: str = Field(description="ok 或 insufficient_evidence")
    order: list[str] = Field(description="违反票数最少的完整顺序;证据不足时为空")
    conflicts: int = Field(description="最少违反票数")
    records: list[RecordConflicts] = Field(description="各有效记录的冲突步骤对")
    valid_records: int = Field(description="参与计算的有效记录数")
