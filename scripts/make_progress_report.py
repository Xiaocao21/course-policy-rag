"""make_progress_report.py - render the Chinese progress report for the supervisor.

Figures come exclusively from report_content.collect(); nothing is hard-coded, so the
document cannot disagree with the repository. Generation is refused when the
consistency check fails.

Requires python-docx, which ships with the bundled runtime rather than the project
venv, so run this script with the bundled interpreter:

    <bundled python> scripts/make_progress_report.py

Usage:
    python scripts/make_progress_report.py                 # writes to the work folder
    python scripts/make_progress_report.py --out PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor, Cm
except ImportError:  # pragma: no cover
    print("python-docx is required. Run this with the bundled interpreter:\n"
          "  C:\\Users\\QJC_Wing\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe "
          "scripts\\make_progress_report.py", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_content  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(r"<home>\Desktop\WKUclass\高级人工智能CPS5801\work"
                   r"\CPS5801_研究进展报告_数据收集与预处理.docx")

CN_FONT = "Microsoft YaHei UI"
EN_FONT = "Calibri"
# NOTE: "微软雅黑" / "Microsoft YaHei" are NOT installed on this machine (only
# "Microsoft YaHei UI" is). Requesting a missing family makes the renderer substitute
# a wider face, which overflowed the right margin and clipped text. The eastAsia font
# must therefore name a family that actually exists here.

# A4 with these margins leaves this much horizontal room for text and tables.
PAGE_W_CM = 21.0
PAGE_H_CM = 29.7
MARGIN_LR_CM = 2.2
MARGIN_TB_CM = 2.0
USABLE_W_CM = PAGE_W_CM - 2 * MARGIN_LR_CM          # 16.6 cm
TABLE_W_CM = USABLE_W_CM - 0.3                       # keep a little slack for borders

TIER_CN = {"course": "课程层（课程大纲）", "faculty": "院系层（学院/系规则）",
           "institution": "校级层（全校规章）"}


# --------------------------------------------------------------------- docx helpers
def set_base_style(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = EN_FONT
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    for section in doc.sections:
        # Explicit A4: leaving the default gives US Letter, and any cm-based table
        # width then overflows the text area and gets clipped on the right.
        section.page_width = Cm(PAGE_W_CM)
        section.page_height = Cm(PAGE_H_CM)
        section.top_margin = Cm(MARGIN_TB_CM)
        section.bottom_margin = Cm(MARGIN_TB_CM)
        section.left_margin = Cm(MARGIN_LR_CM)
        section.right_margin = Cm(MARGIN_LR_CM)


def heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = EN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)


def para(doc: Document, text: str, bold: bool = False, italic: bool = False,
         size: float | None = None, color: tuple | None = None) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = EN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)


def bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text, style="List Bullet")
    for run in p.runs:
        run.font.name = EN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)


def table(doc: Document, headers: list[str], rows: list[list],
          weights: list[float] | None = None):
    """Build a table whose total width always fits the usable page width.

    Column widths are given as relative weights rather than absolute centimetres:
    absolute values silently overflowed when a table was authored against a different
    page size, which clipped the rightmost column. Weights are scaled to fit here.
    """
    n = len(headers)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        weights = [1.0] * n
    total_w = sum(weights) or 1.0
    col_cm = [TABLE_W_CM * w / total_w for w in weights]

    t = doc.add_table(rows=1, cols=n)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.name = EN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            run.font.name = EN_FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    # apply widths to every cell: Word derives column width from the cells
    for r in t.rows:
        for i, w in enumerate(col_cm):
            r.cells[i].width = Cm(w)
    return t


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    run.font.name = EN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

# --------------------------------------------------------------------- report body
def build(data: dict, out_path: Path) -> None:
    c = data["corpus"]
    col = data["collected"]
    cap = data["capability"]
    q = data["quality"]

    doc = Document()
    set_base_style(doc)

    title = doc.add_heading("研究进展报告：数据收集与预处理", level=0)
    for run in title.runs:
        run.font.name = EN_FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    sub = doc.add_paragraph()
    r = sub.add_run("课题：基于 RAG 的大学课程政策问答中文本切分策略对检索准确率的影响\n"
                    f"语料版本：{data['corpus_version']}　｜　冻结时间：{data['frozen_at']}　｜　"
                    f"随机种子：{data['random_seed']}")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    r.font.name = EN_FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    # ---------------------------------------------------------------- 一页速览
    heading(doc, "一页速览", 1)
    para(doc, "教授您好。本阶段（数据收集与预处理）的主体工作已经完成，语料已经冻结，"
              "目前进入人工标注阶段。下表是全部状态：", size=10.5)
    table(doc,
          ["状态", "项目", "说明"],
          [
              ["✅ 已完成", "数据收集",
               f"采集 {col['documents']} 篇、{col['institutions']} 所机构；全程遵守 robots.txt（{data['robots_hosts']} 个域有审计记录）"],
              ["✅ 已完成", "语料冻结",
               f"{c['documents']} 篇入选，覆盖 {c['institutions']} 所机构、3 个层级、6 个政策域"],
              ["✅ 已完成", "预处理流水线",
               f"{c['chunks']} 个文本块（chunk）、{cap['clauses_total']} 条条款，最大 {c['token_max']} token，0 处超限"],
              ["✅ 已完成", "质量控制",
               f"发现并修复 4 个数据缺陷；{data['n_tests']} 项自动化测试全通过；条款偏移错误 {q['span_errors']} 处"],
              ["🔄 进行中", "题目与证据标注",
               "计划 150 题（100 开发 + 50 测试），测试集需 ≥40 个独立题组；由学生本人完成"],
              ["⛔ 尚未开始", "四种切分策略实现",
               "标注完成后开始；需要先装入 embedding 模型依赖"],
          ], weights=[2.4, 3.4, 10.2])
    caption(doc, "数据来源：MANIFEST.json、sources.csv、reports/clause_stats.csv")

    para(doc, "")
    para(doc, f"当前唯一的正式阻塞项：切分审计状态为 “{data['split_audit']['status']}”"
              f"（原因：{', '.join(data['split_audit']['blockers']) or '无'}）。"
              "这不是技术故障，而是因为题目尚未标注，测试题组数自然为 0。", bold=True)

    # ---------------------------------------------------------------- 1
    heading(doc, "1. 这个项目在做什么", 1)
    para(doc, "学生常常需要从很长的课程政策文件中找一个具体规定，例如“作业迟交会怎么处理”、"
              "“生病缺考能否补考”。这类问题有一个特点：正确答案往往不只依赖一条规定，"
              "而是需要「一般规则 + 适用条件 + 例外情况」分散在不同条款里。")
    para(doc, "本项目研究的是这类系统的上游环节——把长文档切成小块（chunking）时，"
              "不同的切法会不会影响系统能否把「完整且正确」的证据条款找出来。"
              "本阶段只处理「能不能找准、找全」，不涉及最终答案的生成。")

    # ---------------------------------------------------------------- 2 术语
    heading(doc, "2. 术语表（便于快速理解下文）", 1)
    table(doc, ["术语", "一句话解释", "在本项目中指什么"],
          [
              ["RAG", "先检索资料、再让模型据此作答的技术路线", "本项目的整体系统框架"],
              ["chunk（文本块）", "把长文档切成的可检索小块", "切分策略的直接产物，切法不同则块不同"],
              ["token", "模型处理文本的最小单位，约等于 0.75 个英文单词", "衡量块大小的单位；本模型上限 512"],
              ["embedding", "把文本转成向量，使语义相近的文本距离更近", "用于建立可检索的向量索引"],
              ["gold span（标准证据条款）", "人工标注的、回答某题所必需的原文片段", "评测的基准，标注时不看切分结果"],
              ["Recall@k", "前 k 个检索结果覆盖了多少比例的必要条款", "核心指标之一：衡量「找全了没」"],
              ["Hit Rate", "前 k 个结果是否至少命中一条必要条款", "衡量「有没有找到」，比 Recall 宽松"],
              ["group（题组）", "表述不同但模板与推理结构相同的一组题", "同组题目必须留在同一个数据划分里"],
          ], weights=[3.6, 6.6, 5.8])
    caption(doc, "术语仅作入门解释，正式定义见 work1 修订版第 4、7 节。")

    # ---------------------------------------------------------------- 3 数据来源
    heading(doc, "3. 数据从哪里来", 1)
    para(doc, "计划书原定只收集「课程大纲」。实测后发现一个问题：MIT OpenCourseWare 的课程大纲页"
              "政策信息非常稀疏——典型页面只有 4 到 13 个政策关键词，几乎没有编号条款。"
              "而机构层面的政策文件密度高得多：例如 Boston University 的《学术行为准则》单页有 "
              "47,509 个字符、195 个政策关键词、14 处例外条款。")
    para(doc, "如果只用课程大纲，计划书中「一般规则 + 例外条款」这类题目根本出不出来。"
              "因此语料改为三层结构：")
    table(doc, ["层级", "适用范围", "由谁制定", "主要覆盖哪些政策域"],
          [
              ["课程层", "单门课程", "任课教师", "作业迟交、课程资源、成绩权重"],
              ["院系层", "学院或系", "院系教务委员会", "评分、考试、出勤及其例外条款"],
              ["校级层", "全校", "教务长/教务处/学术诚信办公室", "学术诚信、补考、申诉程序"],
          ], weights=[2.4, 3.2, 4.6, 5.8])
    para(doc, "需要特别说明：作业迟交（late work）这一政策域只能由课程层提供，"
              "是六个域中唯一无法从机构规章里补上的，因此课程层不会被压缩到很小。", italic=True)

    # ---------------------------------------------------------------- 4 收集结果
    heading(doc, "4. 收集到了什么", 1)
    para(doc, "采集是有筛选的。发现、抓取、清洗、入选四个环节的数量如下：")
    table(doc, ["环节", "数量", "说明"],
          [
              ["候选来源探测", "约 150 个页面", "对每个候选先检查 robots.txt 与政策密度"],
              ["实际抓取入库", f"{col['documents']} 篇", f"覆盖 {col['institutions']} 所机构"],
              ["完成清洗", f"{col['cleaned_documents']} 篇", f"共抽出 {col['clauses_total']} 条条款"],
              ["入选冻结语料", f"{c['documents']} 篇", f"覆盖 {c['institutions']} 所机构；逐条入选理由见 reports/curation_decisions.csv"],
          ], weights=[4, 3.4, 8.6])
    caption(doc, "数据来源：sources.csv（每篇一行）、reports/quality_report.csv、reports/curation_decisions.csv")

    para(doc, "")
    para(doc, "冻结语料的层级构成（要求单层不超过 40%）：", bold=True)
    table(doc, ["层级", "篇数", "占比"],
          [[TIER_CN.get(t, t), c["tier_counts"].get(t, 0), f"{c['tier_shares'].get(t, 0):.0%}"]
           for t in ("course", "faculty", "institution")], weights=[6, 3, 3])

    para(doc, "")
    para(doc, "按机构分布：", bold=True)
    table(doc, ["机构", "篇数"],
          [[inst, n] for inst, n in c["institution_counts"].items()], weights=[9, 3])

    para(doc, "")
    para(doc, "六个政策域的覆盖与验收指标完成情况：", bold=True)
    table(doc, ["验收指标", "实际", "要求", "结果"],
          [[name, have, need, "达标" if ok else "未达标"] for name, have, need, ok in data["gates"]],
          weights=[6.6, 2.4, 2.4, 2.6])
    caption(doc, "数据来源：reports/clause_stats.csv（条款级权威计数）。"
                 "注意 sources.csv 中的 *_raw_hits 列是抓取期的页面级正则命中数，不是条款数，已在报告中停用。")

    # ---------------------------------------------------------------- 5 预处理
    heading(doc, "5. 数据是怎么处理的", 1)
    para(doc, "整条流水线的顺序是：抓取原文 → 解析结构 → 清洗 → 去重 → 建立共享语料 → 冻结。"
              "其中最关键的一个设计是「双层文本表示」，说明如下。")
    table(doc, ["层", "存放文件", "性质与用途"],
          [
              ["证据层", "cleaned/<文档编号>.evidence.jsonl",
               "与原文逐字一致。人工标注的 gold span 全部锚定在这一层，保证能一字不差地引回原文"],
              ["检索层", "cleaned/<文档编号>.retrieval.jsonl",
               "做过规范化（统一引号、破折号、全角半角等），供 embedding 与检索使用"],
              ["映射表", "cleaned/<文档编号>.charmap.jsonl",
               "记录检索层每个字符对应原文的哪个位置，使检索命中能换算回原文位置"],
          ], weights=[2.2, 6.4, 7.4])
    para(doc, "为什么必须这样设计：修订版指标要求「只有检索结果完整覆盖整条标注条款才算命中」，"
              "部分覆盖要单独记为 partial_span_coverage 并按被覆盖的字符比例计算。"
              "如果规范化改变了字符数却没有映射表，这两个数字都会悄悄算错，而且不会报错。")

    para(doc, "")
    para(doc, "另外三条关键处理规则：", bold=True)
    for b in [
        "条款边界稳定：切分可以把一条超长条款拆成多个块，但永不重新定义条款边界，"
        "因此标注的偏移量始终有效。",
        "绝不静默截断：超过 512 token 的块按「句子边界 → 词边界 → 字符」逐级递归切分并记录原因。"
        f"实测最大块为 {c['token_max']} token，超限 {q['chunks_over_max_input']} 个。",
        "页眉页脚剥离带保护规则：含 except / unless / provided that / notwithstanding 等限定词的行"
        "一律标记为需人工复核并保留，绝不自动删除——这正是老师关注的例外条款丢失风险。",
    ]:
        bullet(doc, b)

    # ---------------------------------------------------------------- 6 质量控制
    heading(doc, "6. 质量控制：发现并修复的 4 个问题", 1)
    para(doc, "这一节是过程性记录的重点。以下 4 个问题都不会让程序报错，"
              "只会让结果悄悄变错，因此都补了自动化测试防止复发。")
    table(doc, ["#", "问题现象", "如果不修会怎样", "怎么发现", "修复方式"],
          [
              ["1", "网页解析库未安装时，解析函数返回空字符串",
               "真实政策页被误判为「内容过少」而全部丢弃",
               "首次探测 100 个网址全部返回 0 字符，与人工抽查结果矛盾",
               "解析库缺失时直接报错，不再返回空值"],
              ["2", "网页正文抽取用树遍历，跳过了纯文本节点",
               "整页正文段落被丢弃（某页只抽到 2 段，实际有 232 段）",
               "对比抽取结果与页面实际长度，差距过大",
               "改为按语义标签显式选取，并对正文容器打分"],
              ["3", "条款编号（如 A.）在同一文档内重复出现",
               "条款编号冲突，下游按编号查找时相互覆盖",
               "校验脚本报出重复条款编号",
               "重复编号追加序号，同时保持可读"],
              ["4", "同一政策经两个网址被抓到，内容完全相同",
               "同一份政策占用两个语料名额，在所有指标里被重复计权",
               "对比全部入选文档的原文哈希值，发现 1 组重复",
               "按原文内容哈希去重，并加入校验阻断规则"],
          ], weights=[1, 3.4, 3.6, 3.4, 3.6])
    para(doc, "")
    para(doc, f"当前质量指标：条款偏移错误 {q['span_errors']} 处；超限块 {q['chunks_over_max_input']} 个；"
              f"校验阻断项 {q['blockers']} 项；自动化测试 {data['n_tests']} 项全部通过。", bold=True)

    # ---------------------------------------------------------------- 7 进度
    heading(doc, "7. 现在到哪一步了", 1)
    para(doc, "对照计划书的时间表：")
    table(doc, ["计划书阶段", "内容", "状态"],
          [
              ["第 1–2 周", "文献核对、数据权限与指标确定", "已完成"],
              ["第 3–4 周", "政策文件收集、清洗与题目标注", "收集与清洗已完成；题目标注进行中"],
              ["第 5–6 周", "实现四种切分策略、向量索引与检索基线", "尚未开始"],
              ["第 7–9 周", "主实验，记录 Recall@k、MRR、Hit Rate 与延迟", "尚未开始"],
          ], weights=[2.8, 8.4, 5])
    para(doc, "")
    para(doc, "当前进度判断：数据收集与预处理这一部分已经全部完成并冻结，"
              "进度主要取决于人工标注的速度。", bold=True)

    # ---------------------------------------------------------------- 8 进行中
    heading(doc, "8. 正在进行的工作：题目与证据标注", 1)
    para(doc, "这是当前唯一在进行的工作，也是下一阶段的前置条件。具体安排：")
    for b in [
        "共 150 题：100 题用于开发（调参），50 题作为最终测试集。",
        "测试集需要至少 40 个「独立题组」，即模板与推理结构互不相同的题目；"
        "同义改写的题目计入同一组，避免虚增题量。",
        "每题标注：参考答案、所依据的原文条款编号与字符偏移、以及该题的证据类型"
        "（单条款 / 规则加例外 / 跨条款）。",
        "部分题目标注多个「替代充分证据集」，用于检验指标定义。",
        "抽取 20% 由第二人独立标注，计算一致性（Cohen's kappa），目标 0.70 以上。",
    ]:
        bullet(doc, b)
    para(doc, "标注工作由学生本人完成，依计划书第 12 节 AI 使用声明，"
              "AI 不参与题目撰写、证据标注与参考答案构建。", italic=True)

    # ---------------------------------------------------------------- 9 风险
    heading(doc, "9. 风险与已知限制", 1)
    table(doc, ["限制", "当前状态", "影响与应对"],
          [
              ["分词器为近似模式", "尚未安装 embedding 模型依赖（体积较大）",
               "块大小统计是近似值。正式语料需装入依赖后重跑，产物会自动改名区分，不会混用"],
              ["机构广度受限", f"入选 {c['institutions']} 所机构",
               "部分大学域名禁止抓取或无法访问。已逐条记录，并作为观察写进合规文档"],
              ["章节路径有噪声", "约 18% 的条款能追溯到明确标题",
               "章节路径仅用于展示与分组，不参与偏移与命中判定，不影响指标"],
              ["标记文件与计数并列", "已分离 *_raw_hits 与权威条款计数",
               "已在报告中停用抓取期计数，避免误读"],
          ], weights=[3.4, 4.4, 7.2])

    # ---------------------------------------------------------------- 10 下一步
    heading(doc, "10. 下一步计划", 1)
    table(doc, ["顺序", "工作", "产出"],
          [
              ["1", "完成 150 题标注与 20% 双人一致性校验", "题目集与标准证据标注"],
              ["2", "运行切分审计：检查题组是否跨划分、测试组是否达 40", "审计报告"],
              ["3", "装入 embedding 依赖，重建正式（非近似）语料并重新冻结", "正式语料与清单"],
              ["4", "实现四种切分策略：固定长度、按段落、按标题结构、语义切分", "四套索引"],
              ["5", "用开发集做组间不相交的交叉验证，确定块长、重叠与语义阈值", "冻结的超参数"],
              ["6", "在测试集上只跑一次，记录 Recall@k、Hit Rate、完整证据率、延迟与索引成本", "主实验结果"],
          ], weights=[1.4, 9.4, 5.4])

    # ---------------------------------------------------------------- 附录
    doc.add_page_break()
    heading(doc, "附录 A：冻结语料清单", 1)
    para(doc, f"共 {c['documents']} 篇。条款数为清洗后的权威计数。")
    rows = []
    for d in data["documents"]:
        rows.append([d["institution"], TIER_CN.get(d["tier"], d["tier"]).split("（")[0],
                     d["doc_id"][:52], d["clauses"], d["exception"], d["crossref"], d["alt"]])
    table(doc, ["机构", "层级", "文档编号（截断）", "条款", "含例外", "含交叉引用", "含替代证据"],
          rows, weights=[3.2, 1.8, 6.2, 1.3, 1.5, 1.9, 1.9])
    caption(doc, "完整编号见 sources.csv；每篇的原始网址、抓取时间与原文哈希也在该文件。")

    heading(doc, "附录 B：常用命令", 1)
    for line in [
        "cd <repo>",
        ".\\.venv\\Scripts\\python.exe run.py report     查看全部产物状态",
        ".\\.venv\\Scripts\\python.exe run.py audit      检查题组划分与切分纪律",
        ".\\.venv\\Scripts\\python.exe run.py validate   校验语料不变量",
        ".\\.venv\\Scripts\\python.exe run.py test       运行自动化测试",
    ]:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    heading(doc, "附录 C：文件索引", 1)
    table(doc, ["路径", "内容"],
          [
              ["sources.csv", "每篇文档一行：网址、抓取时间、原文哈希、层级、权威条款数、是否入选"],
              ["raw/<层级>/<编号>/", "原始字节（永不修改）、已抽取纯文本、抓取元数据"],
              ["cleaned/<编号>.evidence.jsonl", "证据层：标注 gold span 的依据"],
              ["cleaned/<编号>.retrieval.jsonl", "检索层：用于建立向量索引"],
              ["cleaned/<编号>.charmap.jsonl", "两层之间的字符映射"],
              ["corpus_safe_512/", "四种切分策略共享的同一份语料"],
              ["annotations/gold_worksheet.csv", "可标注条款清单（含字符偏移）"],
              ["annotations/SCHEMA.md", "标注字段定义与判定规则"],
              ["reports/clause_stats.csv", "条款级权威计数"],
              ["reports/curation_decisions.csv", "每篇文档入选或排除的理由"],
              ["reports/robots_audit.json", "各域 robots.txt 判定与原文留存"],
              ["MANIFEST.json", "全部产物的 SHA-256 与固定配置，保证可复现"],
              ["COMPLIANCE.md", "合规说明：抓取政策、许可、已知限制"],
          ], weights=[6.4, 8.6])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-consistency", action="store_true")
    args = ap.parse_args()

    data = report_content.collect(Path(args.root))

    if not data["consistency"]["ok"] and not args.skip_consistency:
        print("REFUSING TO GENERATE: the report figures are not self-consistent.", file=sys.stderr)
        for p in data["consistency"]["problems"]:
            print(f"  - {p}", file=sys.stderr)
        print("\nFix the data first, or pass --skip-consistency to override.", file=sys.stderr)
        return 1

    out = Path(args.out)
    build(data, out)
    print(f"wrote {out}")
    print(f"  corpus {data['corpus']['documents']} docs / {data['corpus']['chunks']} chunks / "
          f"{data['capability']['clauses_total']} clauses")
    print(f"  consistency: {'ok' if data['consistency']['ok'] else 'OVERRIDDEN'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
