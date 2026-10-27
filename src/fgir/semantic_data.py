"""Source-bound semantic records, honest review gates and immutable diagnostics.

No language model is called here. Structural validity cannot certify that a
caption is true of an image; that assertion always needs a recorded review.
"""

from collections import Counter
import hashlib
from pathlib import Path

from .contracts import image_path, load_manifest, text
from .data import file_digest
from .files import new_directory, read_json, read_jsonl, write_json, write_jsonl


PHENOMENA = ("attribute_binding", "relation_direction", "paraphrase", "false_addition")
VARIANT_TYPES = ("minimal_edit", "paraphrase", "true_addition", "false_addition")
STATUSES = ("draft", "annotated", "auto_verified", "reviewed", "rejected")


def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def objects(value, label):
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a list of objects")
    return value


def span_text(source, span, expected, label, allow_empty=False):
    if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
            or not 0 <= span[0] <= span[1] <= len(source) or (not allow_empty and span[0] == span[1])):
        raise ValueError(f"invalid Unicode span: {label}")
    if source[span[0]:span[1]] != expected:
        raise ValueError(f"span does not recover original text: {label}")


def normalized(value):
    return " ".join(value.casefold().split())


def parsing_sets(record, source, label="semantic parsing"):
    """Shared validation and ID alignment for drafts, frozen gold and predictions.

    Canonical mentions use source span plus normalized head; attribute ownership
    and directed relations use those mentions rather than parser-local IDs.
    """
    if not isinstance(record, dict):
        raise ValueError(f"{label}: expected a parsing object")
    entities, spans, canonical, attrs, relations = {}, set(), set(), set(), set()
    ids = set()
    for entity in objects(record.get("entities"), f"{label} entities"):
        eid = text(entity.get("id"), f"{label} entity id")
        if eid in ids or eid == "sentence":
            raise ValueError(f"{label}: duplicate/reserved semantic ID")
        ids.add(eid)
        head = text(entity.get("head"), f"{label} entity head")
        span_text(source, entity.get("span"), head, f"{label} {eid}")
        span = tuple(entity["span"])
        key = (*span, normalized(head))
        if key in canonical:
            raise ValueError(f"{label}: duplicate entity mention")
        spans.add(span)
        canonical.add(key)
        entities[eid] = key
        for attr in objects(entity.get("attributes"), f"{label} attributes"):
            if attr.get("entity_id") != eid:
                raise ValueError(f"{label}: attribute owner mismatch")
            value = text(attr.get("value"), f"{label} attribute value")
            span_text(source, attr.get("span"), value, f"{label} {eid}")
            item = (key, normalized(text(attr.get("name"), f"{label} attribute name")), normalized(value))
            if item in attrs:
                raise ValueError(f"{label}: duplicate attribute")
            attrs.add(item)
    for relation in objects(record.get("relations"), f"{label} relations"):
        rid = text(relation.get("id"), f"{label} relation id")
        if rid in ids or rid == "sentence":
            raise ValueError(f"{label}: duplicate/reserved semantic ID")
        ids.add(rid)
        subject = text(relation.get("subject"), f"{label} subject")
        obj = text(relation.get("object"), f"{label} object")
        if subject not in entities or obj not in entities:
            raise ValueError(f"{label}: relation references missing entity")
        predicate = text(relation.get("predicate"), f"{label} predicate")
        span_text(source, relation.get("span"), predicate, f"{label} {rid}")
        item = (entities[subject], normalized(predicate), entities[obj])
        if item in relations:
            raise ValueError(f"{label}: duplicate relation")
        relations.add(item)
    return {"entity_span": spans, "entity_normalized": canonical,
            "attribute_binding": attrs, "directed_relation": relations}


def source_inventory_image(context, image_id):
    """Require canonical inventory metadata, including in a frozen context."""
    inventory = context.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("source context must bind inventory; re-freeze older semantic protocols")
    item = inventory.get(image_id)
    image = context["images"].get(image_id)
    if not isinstance(item, dict) or image is None or any(item.get(k) != image.get(k) for k in ("image_id", "path", "split")):
        raise ValueError(f"source inventory image/path/split mismatch: {image_id}")
    digest(item.get("file_sha256"), "source inventory image hash")
    return item


def verify_source_image(context, image_id, data_root):
    item = source_inventory_image(context, image_id)
    actual = file_digest(image_path(data_root, item["path"]))
    if actual != item["file_sha256"]:
        raise ValueError(f"image file hash mismatch (different image bytes from source inventory): {image_id}")
    return actual


def review_check(review, purpose, label, visual=False):
    if not isinstance(review, dict) or review.get("status") not in STATUSES:
        raise ValueError(f"invalid review status: {label}")
    if not isinstance(review.get("disputes", []), list):
        raise ValueError(f"disputes must be a list: {label}")
    if review.get("disputes", []) and purpose != "draft":
        raise ValueError(f"unresolved disputes: {label}")
    status = review["status"]
    if status == "rejected":
        text(review.get("notes"), f"rejection reason: {label}")
    if status in ("annotated", "auto_verified", "reviewed"):
        author = text(review.get("annotator"), f"annotator: {label}")
        kind = review.get("annotator_kind")
        if kind not in ("human", "assistant", "automatic"):
            raise ValueError(f"declare annotator_kind: {label}")
        text(review.get("notes"), f"review notes: {label}")
        if status == "auto_verified" and kind != "automatic":
            raise ValueError(f"auto_verified requires automatic provenance: {label}")
        if visual:
            text(review.get("image_evidence"), f"image evidence: {label}")
            digest(review.get("image_sha256"), f"review image: {label}")
        if status == "reviewed":
            reviewer = text(review.get("reviewer"), f"reviewer: {label}")
            if reviewer.strip().casefold() == author.strip().casefold() or review.get("reviewer_kind") != "human":
                raise ValueError(f"independent human reviewer required: {label}")
            if kind != "human":
                raise ValueError(f"evaluation gold requires a human initial annotator: {label}")
            text(review.get("review_notes"), f"independent review notes: {label}")
    if purpose == "eval" and status != "reviewed":
        raise ValueError(f"not independently reviewed: {label}")
    if purpose == "train" and status not in ("annotated", "auto_verified", "reviewed"):
        raise ValueError(f"unverified training supervision: {label}")


def source_context(bundle):
    """Check authoritative files and resolve retained known scene groups."""
    bundle = Path(bundle)
    dataset = read_json(bundle / "dataset.json")
    if dataset.get("schema_version") != "fgir-dataset-v1":
        raise ValueError("unsupported source dataset")
    data_kind = dataset.get("data_kind", "real")
    if data_kind not in ("real", "synthetic"):
        raise ValueError("invalid source data_kind")
    names = ["manifest.jsonl", "inventory.jsonl", "duplicate-review.json", *[f"{s}/pairs.jsonl" for s in ("train", "val", "test")]]
    for name in names:
        digest(dataset.get("files", {}).get(name), f"source file binding: {name}")
        if file_digest(bundle / name) != dataset.get("files", {}).get(name):
            raise ValueError(f"source file hash mismatch: {name}")
    if (dataset.get("image_inventory_sha256") is not None
            and dataset["image_inventory_sha256"] != dataset["files"]["inventory.jsonl"]):
        raise ValueError("source inventory digest fields disagree")
    manifest = load_manifest(bundle / "manifest.jsonl")
    images = {row["image_id"]: row for row in manifest}
    inventory = {}
    for item in read_jsonl(bundle / "inventory.jsonl"):
        image_id = text(item.get("image_id"), "inventory image_id")
        if image_id in inventory:
            raise ValueError(f"duplicate source inventory image: {image_id}")
        inventory[image_id] = item
        source_inventory_image({"images": images, "inventory": inventory}, image_id)
    if set(inventory) != set(images):
        raise ValueError("source inventory must cover exactly the manifest images")
    pairs = {}
    for split in ("train", "val", "test"):
        for row in read_jsonl(bundle / split / "pairs.jsonl"):
            image = images.get(row.get("image_id"))
            if image is None or row.get("split") != split or image["split"] != split:
                raise ValueError("source pairs disagree with manifest split")
            query_id = row.get("query_id")
            expected = {f"{image['image_id']}#{i}": caption for i, caption in enumerate(image["captions"])}
            if query_id in pairs or expected.get(query_id) != row.get("text") or row.get("path") != image["path"]:
                raise ValueError("source pairs disagree with original caption or path")
            pairs[query_id] = row
    if len(pairs) != sum(len(row["captions"]) for row in manifest):
        raise ValueError("source pairs are incomplete")
    parent = {image_id: image_id for image_id in images}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for pair in read_json(bundle / "duplicate-review.json")["pairs"]:
        a, b = pair["left"], pair["right"]
        if pair["decision"] in ("identical", "same_scene") and a in parent and b in parent:
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)
    scenes = {image_id: find(image_id) for image_id in images}
    splits = {}
    for image_id, scene in scenes.items():
        if scene in splits and splits[scene] != images[image_id]["split"]:
            raise ValueError("known source scene crosses splits")
        splits[scene] = images[image_id]["split"]
    return {"images": images, "inventory": inventory, "pairs": pairs, "scenes": scenes,
            "data_kind": data_kind,
            "manifest_sha256": file_digest(bundle / "manifest.jsonl"),
            "source_files": {name: file_digest(bundle / name) for name in names}}


def validate_record(row, context, purpose="draft"):
    if purpose not in ("draft", "train", "eval"):
        raise ValueError("purpose must be draft, train or eval")
    if row.get("schema_version") != "fgir-semantic-v1":
        raise ValueError("unsupported semantic record")
    sample_id = text(row.get("sample_id"), "sample_id")
    pair = context["pairs"].get(text(row.get("source_query_id"), "source_query_id"))
    if pair is None or any(row.get(a) != pair[b] for a, b in (
            ("source_image_id", "image_id"), ("split", "split"), ("text", "text"))):
        raise ValueError(f"source caption/image/split mismatch: {sample_id}")
    if row.get("group_id") != pair["image_id"] or row.get("scene_group_id") != context["scenes"][pair["image_id"]]:
        raise ValueError("source image/scene grouping mismatch")
    if purpose == "train" and row["split"] != "train":
        raise ValueError("training input must come only from train")
    if purpose == "eval" and row["split"] not in ("val", "test"):
        raise ValueError("diagnostics must come from val or test")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source_manifest_sha256") != context["manifest_sha256"]:
        raise ValueError("semantic source manifest hash mismatch")
    generator = provenance.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("generator provenance required")
    text(generator.get("name"), "generator name")
    for key in ("code_revision",):
        text(provenance.get(key), key)
    if type(provenance.get("seed")) is not int:
        raise ValueError("generator seed must be an integer")
    for key in ("weights_sha256", "prompt_sha256"):
        value = generator.get(key)
        if value is not None:
            digest(value, key)
        elif generator["name"] not in ("source-only", "human"):
            raise ValueError(f"model generator must record {key}")
    review_check(row.get("review"), purpose, sample_id, visual=True)
    if row.get("source_visual_label") not in ("supported", "uncertain"):
        raise ValueError("source_visual_label must be supported or uncertain")
    if purpose != "draft" and row["source_visual_label"] != "supported":
        raise ValueError("source positive needs image support")
    parsing_sets(row, row["text"], f"gold parsing {sample_id}")
    entities = {entity["id"]: entity for entity in row["entities"]}
    ids = set(entities) | {relation["id"] for relation in row["relations"]}
    variants = {}
    for variant in objects(row.get("variants"), "variants"):
        vid = text(variant.get("id"), "variant id")
        if vid in variants or vid in ids or vid in ("sentence", "original"):
            raise ValueError("duplicate/reserved variant ID")
        variants[vid] = variant
        kind = variant.get("type")
        if kind not in VARIANT_TYPES or text(variant.get("target_unit_id"), "variant target") not in ids | {"sentence"}:
            raise ValueError("unknown variant type or target")
        expected = "equivalent" if kind == "paraphrase" else "changed"
        if variant.get("semantic_relation") != expected:
            raise ValueError("variant semantic relation disagrees with type")
        phenomenon = variant.get("phenomenon")
        if (phenomenon not in PHENOMENA or (kind == "minimal_edit" and phenomenon not in PHENOMENA[:2])
                or (kind == "paraphrase" and phenomenon != "paraphrase")
                or (kind.endswith("addition") and phenomenon != "false_addition")):
            raise ValueError("variant phenomenon disagrees with type")
        edit = variant.get("edit")
        if not isinstance(edit, dict):
            raise ValueError("edit evidence required")
        before, after = edit.get("before"), edit.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("edit before/after must be strings")
        text(edit.get("field"), "edited field")
        span_text(row["text"], edit.get("span"), before, vid, allow_empty=True)
        start, end = edit["span"]
        if row["text"][:start] + after + row["text"][end:] != text(variant.get("text"), "variant text"):
            raise ValueError("variant text does not match declared edit")
        if variant["text"] == row["text"]:
            raise ValueError("variant must change text")
        if kind.endswith("addition") and (start != len(row["text"]) or end != start or before or not after.strip()):
            raise ValueError("addition must append a nonempty detail at end")
        visual = variant.get("visual_label")
        if visual not in ("supported", "contradicted", "uncertain", "not_applicable"):
            raise ValueError("invalid visual_label")
        review_check(variant.get("review"), purpose, vid, visual=True)
        if purpose != "draft":
            required_label = "supported" if kind in ("paraphrase", "true_addition") else "contradicted"
            if visual != required_label or variant.get("naturalness") != "acceptable":
                raise ValueError("verified variant needs matching visual label and acceptable naturalness")
            if kind.endswith("addition"):
                text(variant.get("control_pair_id"), "addition control pair id")
    for unit in objects(row.get("units"), "units"):
        uid = text(unit.get("id"), "unit id")
        if uid in ids or uid in variants or uid == "sentence":
            raise ValueError("duplicate/reserved unit ID")
        ids.add(uid)
        if unit.get("type") not in ("entity", "relation"):
            raise ValueError("unknown component type")
        refs = unit.get("entity_ids")
        if (not isinstance(refs, list) or not refs or any(not isinstance(x, str) for x in refs)
                or len(set(refs)) != len(refs) or not set(refs) <= entities.keys()):
            raise ValueError("component entity references invalid")
        span_text(row["text"], unit.get("span"), text(unit.get("text"), "component text"), uid)
        text(unit.get("foil"), "component foil")
        variant = variants.get(text(unit.get("variant_id"), "component variant_id"))
        if variant is None or variant["type"] != "minimal_edit":
            raise ValueError("component must link to a full-sentence minimal edit")
        # The same edit must occur inside the anchored component, not a second foil.
        a, b = unit["span"]
        c, d = variant["edit"]["span"]
        if not a <= c <= d <= b or row["text"][a:c] + variant["edit"]["after"] + row["text"][d:b] != unit["foil"]:
            raise ValueError("component foil disagrees with linked edit")
        review_check(unit.get("review"), purpose, uid, visual=True)
    if purpose != "draft":
        image_hash = row["review"]["image_sha256"]
        if any(item["review"]["image_sha256"] != image_hash for item in [*variants.values(), *row["units"]]):
            raise ValueError("review image hashes disagree within source group")
    canonical_hash = source_inventory_image(context, row["source_image_id"])["file_sha256"]
    for review in [row["review"], *[item["review"] for item in row["variants"] + row["units"]]]:
        if review.get("image_sha256") is not None and review["image_sha256"] != canonical_hash:
            raise ValueError(f"review image hash differs from source inventory: {sample_id}")
    return row


def validate_semantics(samples, bundle, purpose="draft", data_root=None):
    context = source_context(bundle)
    rows = read_jsonl(samples)
    seen = set()
    for row in rows:
        validate_record(row, context, purpose)
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])
        if data_root is not None:
            actual = verify_source_image(context, row["source_image_id"], data_root)
            reviews = [row["review"], *[x["review"] for x in row["variants"] + row["units"]]]
            if any(r.get("image_sha256") is not None and r["image_sha256"] != actual for r in reviews):
                raise ValueError("review evidence belongs to different image bytes")
    return {"schema_version": "fgir-semantic-audit-v1", "purpose": purpose, "data_kind": context["data_kind"], "sample_count": len(rows),
            "source_image_count": len({r["source_image_id"] for r in rows}),
            "splits": dict(Counter(r["split"] for r in rows)),
            "review_status": dict(Counter(r["review"]["status"] for r in rows)),
            "source_visual_labels": dict(Counter(r["source_visual_label"] for r in rows)),
            "records_without_entities": sum(not r["entities"] for r in rows),
            "records_without_variants": sum(not r["variants"] for r in rows),
            "unit_count": sum(len(r["units"]) for r in rows),
            "variant_types": dict(Counter(v["type"] for r in rows for v in r["variants"])),
            "variant_visual_labels": dict(Counter(v["visual_label"] for r in rows for v in r["variants"])),
            "phenomena": dict(Counter(v["phenomenon"] for r in rows for v in r["variants"])),
            "samples_sha256": file_digest(samples), "source_manifest_sha256": context["manifest_sha256"],
            "image_bytes_checked": data_root is not None,
            "limitation": "Structural checks do not establish semantic equivalence or visual truth. Review identities are attestations, not authentication."}


def prepare_semantic_pilot(bundle, data_root, output, count=30, seed=42, code_revision="working-tree"):
    if type(count) is not int or count <= 0 or type(seed) is not int:
        raise ValueError("count must be positive and seed an integer")
    context = source_context(bundle)
    pool = [r for r in context["images"].values() if r["split"] == "val"]
    if count > len(pool):
        raise ValueError("pilot count exceeds val source images")

    def order(key):
        return hashlib.sha256(f"semantic-pilot-v1:{seed}:{key}".encode()).hexdigest()

    selected = sorted(pool, key=lambda r: order(r["image_id"]))[:count]
    rows = []
    for i, image in enumerate(selected):
        pair = min((context["pairs"][f"{image['image_id']}#{j}"] for j in range(len(image["captions"]))),
                   key=lambda r: order(r["query_id"]))
        image_hash = verify_source_image(context, image["image_id"], data_root)
        rows.append({"schema_version": "fgir-semantic-v1", "sample_id": f"semantic:{pair['query_id']}",
                     "group_id": image["image_id"], "scene_group_id": context["scenes"][image["image_id"]],
                     "source_image_id": image["image_id"], "source_query_id": pair["query_id"], "split": "val",
                     "text": pair["text"], "source_visual_label": "uncertain", "entities": [], "relations": [],
                     "variants": [], "units": [], "assigned_to": "ABCD"[i % 4],
                     "independent_reviewer": {"A": "B", "B": "A", "C": "D", "D": "C"}["ABCD"[i % 4]],
                     "provenance": {"source_manifest_sha256": context["manifest_sha256"],
                                    "generator": {"name": "source-only", "weights_sha256": None, "prompt_sha256": None},
                                    "seed": seed, "code_revision": code_revision},
                     "review": {"status": "draft", "image_sha256": image_hash, "disputes": []}})
    with new_directory(output) as target:
        write_jsonl(target / "samples.work.jsonl", rows)
        audit = validate_semantics(target / "samples.work.jsonl", bundle, data_root=data_root)
        write_json(target / "selection.json", {"schema_version": "fgir-semantic-pilot-v1", "seed": seed,
                   "count": count, "split": "val", "selection_rule": "SHA256 order of seed and image ID, then caption ID; no model scores or semantic screening",
                   "test_access": "No test captions/images selected; source metadata validated across splits. Historical test exposure remains disclosed in docs/experiments.md.",
                   "source_files": context["source_files"], "assignments": dict(Counter(r["assigned_to"] for r in rows)),
                   "samples_sha256": file_digest(target / "samples.work.jsonl"), "audit": audit})
    return {"output": str(output), **audit}


def freeze_semantics(samples, bundle, data_root, output, data_kind, test_access_notes, recall_ks=(1, 5, 10)):
    if data_kind not in ("real", "synthetic"):
        raise ValueError("data_kind must be real or synthetic")
    text(test_access_notes, "test access disclosure")
    audit = validate_semantics(samples, bundle, "eval", data_root)
    if len(audit["splits"]) != 1:
        raise ValueError("freeze val and test separately")
    ks = list(recall_ks)
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(ks) != len(set(ks)):
        raise ValueError("recall_ks must be unique positive integers")
    rows, context = read_jsonl(samples), source_context(bundle)
    if context["data_kind"] != data_kind:
        raise ValueError("cannot relabel source data_kind when freezing semantic diagnostics")
    if not any(row["variants"] for row in rows):
        raise ValueError("diagnostic protocol needs reviewed variants")
    split = rows[0]["split"]
    candidates = sorted(k for k, v in context["images"].items() if v["split"] == split)
    with new_directory(output) as target:
        write_jsonl(target / "samples.jsonl", rows)
        write_json(target / "candidates.json", candidates)
        write_json(target / "audit.json", audit)
        needed_queries = {r["source_query_id"] for r in rows}
        needed_images = {r["source_image_id"] for r in rows}
        bound_context = {**context, "pairs": {k: v for k, v in context["pairs"].items() if k in needed_queries},
                         "images": {k: v for k, v in context["images"].items() if k in needed_images},
                         "inventory": {k: v for k, v in context["inventory"].items() if k in needed_images},
                         "scenes": {k: v for k, v in context["scenes"].items() if k in needed_images}}
        write_json(target / "source-context.json", bound_context)
        protocol = {"schema_version": "fgir-semantic-eval-v1", "data_kind": data_kind, "split": split,
                    "source_manifest_sha256": context["manifest_sha256"], "independently_reviewed": True,
                    "sample_count": len(rows), "source_image_count": audit["source_image_count"],
                    "candidate_count": len(candidates), "recall_ks": ks, "tie_credit": 0.5,
                    "aggregation": "source_image_macro", "normalization": "unicode-casefold-whitespace-v1",
                    "test_access_notes": test_access_notes,
                    "files": {name: {"path": name, "sha256": file_digest(target / name)} for name in
                              ("samples.jsonl", "candidates.json", "audit.json", "source-context.json")}}
        write_json(target / "protocol.json", protocol)
    return {"output": str(output), "protocol_sha256": file_digest(Path(output) / "protocol.json"), **protocol}


def apply_semantic_source_checks(samples, bundle, data_root, checks_path, output):
    """Import explicitly assistant-only caption checks without overwriting work.

    These checks do not supply a parse, a variant, or independent human gold.
    Uncertain sources remain uncertain and cannot pass the evaluation gate.
    """
    validate_semantics(samples, bundle, data_root=data_root)
    checks = read_json(checks_path)
    if not isinstance(checks, dict) or checks.get("schema_version") != "fgir-semantic-source-checks-v1":
        raise ValueError("unsupported semantic source checks")
    context = source_context(bundle)
    if checks.get("source_manifest_sha256") != context["manifest_sha256"]:
        raise ValueError("source checks manifest mismatch")
    rows = read_jsonl(samples)
    by_id = {r["sample_id"]: r for r in rows}
    seen = set()
    for check in objects(checks.get("checks"), "source checks"):
        sid = check.get("sample_id")
        if sid not in by_id or sid in seen:
            raise ValueError("unknown or duplicate source check")
        seen.add(sid)
        row = by_id[sid]
        if row["split"] != "val" or row["review"]["status"] != "draft":
            raise ValueError("source checks only annotate untouched val drafts")
        if hashlib.sha256(row["text"].encode()).hexdigest() != check.get("text_sha256"):
            raise ValueError("source check caption hash mismatch")
        actual = verify_source_image(context, row["source_image_id"], data_root)
        if actual != check.get("image_sha256"):
            raise ValueError("source check image hash mismatch")
        if check.get("source_visual_label") not in ("supported", "uncertain"):
            raise ValueError("source check cannot convert uncertainty to a negative")
        text(check.get("image_evidence"), "source image evidence")
        row["source_visual_label"] = check["source_visual_label"]
        row["review"] = {"status": "annotated", "annotator": "A-assisted (Codex visual inspection)",
                         "annotator_kind": "assistant", "image_sha256": actual, "image_evidence": check["image_evidence"],
                         "notes": "Source caption visual precheck only. Parse/variant gold and independent human annotation/review are outstanding.",
                         "disputes": check.get("disputes", [])}
    if not seen:
        raise ValueError("empty source checks")
    with new_directory(output) as target:
        write_jsonl(target / "samples.work.jsonl", rows)
        audit = validate_semantics(target / "samples.work.jsonl", bundle, data_root=data_root)
        write_json(target / "receipt.json", {"source_samples_sha256": file_digest(samples),
                   "source_checks_sha256": file_digest(checks_path), "assisted_source_checks": len(seen), "audit": audit})
    return {"output": str(output), "assisted_source_checks": len(seen), **audit}
