"""Training-negative audits, latency summaries and local visual case sheets."""

from collections import defaultdict
import html
import math
from pathlib import Path

from .annotations import thumbnail
from .contracts import image_path, load_manifest, text
from .evaluation import _percentile
from .files import new_directory, read_jsonl
from .metrics import _validate_ids


def validate_negatives(manifest, negatives):
    images = load_manifest(manifest)
    train = {row["image_id"] for row in images if row["split"] == "train"}
    queries = {f"{row['image_id']}#{i}": row["image_id"] for row in images if row["split"] == "train"
               for i in range(len(row["captions"]))}
    rows = read_jsonl(negatives)
    seen, negative_count, reviewed_count = set(), 0, 0
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id not in queries or query_id in seen:
            raise ValueError("mined query must be a unique training query ID")
        seen.add(query_id)
        positive, negative = row.get("positive_image_ids"), row.get("negative_image_ids")
        _validate_ids(positive, "positive IDs", require_nonempty=True)
        _validate_ids(negative, "negative IDs", require_nonempty=True)
        if not (set(positive) | set(negative)) <= train:
            raise ValueError(f"negative mining contains non-training images: {query_id}")
        if queries[query_id] not in positive or set(positive) & set(negative):
            raise ValueError(f"known positives are missing or used as negatives: {query_id}")
        text(row.get("mining_model"), "mining_model")
        if type(row.get("reviewed")) is not bool:
            raise ValueError("negative records must explicitly declare reviewed true/false")
        if row["reviewed"]:
            text(row.get("reviewer"), "negative reviewer")
            text(row.get("review_notes"), "negative review notes")
            reviewed_count += 1
        negative_count += len(negative)
    return {"ok": True, "query_count": len(rows), "negative_count": negative_count,
            "reviewed_query_count": reviewed_count, "reviewed_query_fraction": reviewed_count / len(rows),
            "training_query_coverage": len(rows) / len(queries),
            "limitation": "Structural checks cannot establish that unlabelled similar images are true negatives. Review sampling and false-negative estimates are still required."}


def summarize_latency(path):
    groups = defaultdict(list)
    excluded = defaultdict(int)
    for row in read_jsonl(path):
        stage, device = row.get("stage"), row.get("device")
        if stage not in ("index_build", "text_encoding", "vector_sort", "end_to_end", "cold_start"):
            raise ValueError("unknown latency stage")
        text(device, "device")
        for key in ("gallery_size", "top_k", "batch_size"):
            if type(row.get(key)) is not int or row[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        text(row.get("cache_state"), "cache_state")
        for key in ("warmup", "cold_start"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"{key} must be explicitly true/false")
        milliseconds = row.get("milliseconds")
        if type(milliseconds) not in (int, float) or not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("latency must be finite non-negative milliseconds")
        if device.lower().startswith(("cuda", "gpu", "mps")) and row.get("synchronized") is not True:
            raise ValueError("accelerator timings must explicitly record device synchronization")
        if row["warmup"]:
            excluded["warmup"] += 1
            continue
        cold_start = row["cold_start"] or stage == "cold_start"
        if cold_start:
            excluded["cold_start"] += 1
        key = (stage, device, row["gallery_size"], row["top_k"], row["batch_size"], row["cache_state"], cold_start)
        groups[key].append(milliseconds)
    return {"quantile_definition": "linear_interpolation_at_(n-1)*p", "excluded_from_steady_state": dict(excluded),
            "groups": [dict(zip(("stage", "device", "gallery_size", "top_k", "batch_size", "cache_state", "cold_start"), key),
                            sample_count=len(values), p50_ms=_percentile(values, .5), p95_ms=_percentile(values, .95),
                            min_ms=min(values), max_ms=max(values)) for key, values in sorted(groups.items())]}


def render_cases(manifest, cases, data_root, output):
    images = {row["image_id"]: row for row in load_manifest(manifest)}
    rows = read_jsonl(cases)
    with new_directory(output) as target:
        parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>检索案例复核</title>",
                 "<style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f4f6fa}section{background:white;padding:20px;margin:20px 0}.strip{display:flex;flex-wrap:wrap;gap:8px}figure{width:155px;margin:5px}img{width:150px;height:130px;object-fit:contain}figcaption{font-size:11px;overflow-wrap:anywhere}textarea{width:95%;min-height:60px}h2{font-size:19px}</style>",
                 "<h1>真实结果案例复核</h1><p>成功与失败、改善与退化均需保留。模型错误与标注不完整必须区分；下列原因初始为空，不能将自动选例当作人工分析。</p>",
                 "<button onclick='download()'>导出复核记录 JSONL</button>"]
        for i, row in enumerate(rows):
            parts.append(f"<section><h2>{html.escape(row['query_id'])}: {html.escape(row['text'])}</h2><p>{html.escape(str(row.get('categories', [])))} / {html.escape(row.get('outcome', ''))}</p>")
            for label, key in (("已标注相关图片", "relevant_image_ids"), ("基线 Top-K", "baseline_ranking"), ("比较方法 Top-K", "candidate_ranking")):
                parts.append(f"<h3>{label}</h3><div class='strip'>")
                for rank, image_id in enumerate(row.get(key, [])[:10], 1):
                    if image_id not in images:
                        raise ValueError(f"case contains unknown image ID: {image_id}")
                    src = thumbnail(image_path(data_root, images[image_id]["path"]))
                    parts.append(f"<figure><img src='{src}' alt='{html.escape(image_id)}'><figcaption>{rank}. {html.escape(image_id)}</figcaption></figure>")
                parts.append("</div>")
            parts.append(f"<label>错误类型 <select data-index='{i}' data-field='error_type'><option value=''>待复核</option><option>text_ambiguity</option><option>color</option><option>count</option><option>action</option><option>relation</option><option>incomplete_annotation</option><option>other</option><option>success</option></select></label><p><textarea data-index='{i}' data-field='analysis_notes' placeholder='记录看图证据、原因、标注争议和复核人'></textarea></p></section>")
        import json
        payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
        parts.extend([f"<script type='application/json' id='rows'>{payload}</script>",
                      "<script>const rows=JSON.parse(document.getElementById('rows').textContent);document.querySelectorAll('[data-field]').forEach(e=>{e.value=rows[Number(e.dataset.index)][e.dataset.field]||'';e.onchange=()=>{const r=rows[Number(e.dataset.index)];r[e.dataset.field]=e.value;r.analysis_status='edited_requires_review'}});function download(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>JSON.stringify(r)).join('\\n')+'\\n'],{type:'application/x-ndjson'}));a.download='cases.review.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}</script></html>"])
        (target / "review.html").write_text("\n".join(parts), encoding="utf-8")
    return {"case_count": len(rows), "output": str(output)}
