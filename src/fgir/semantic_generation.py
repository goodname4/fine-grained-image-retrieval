"""B-V2-01: local language-model candidate generation for semantic records.

Proposes entity/attribute/relation parses and variants (paraphrase, single-
condition minimal edits, paired true/false additions) for each source caption.
Every proposed phrase is anchored to exact substrings of the original caption;
unanchorable proposals are excluded with recorded reasons. All labels stay
draft: visual truth, semantic equivalence and object binding remain for human
review and are never certified here.

Heavy model imports are deferred so the standard-library CLI stays cheap.
"""

import hashlib
import os
import random
import re
from pathlib import Path

from .data import file_digest
from .files import new_directory, read_json, read_jsonl, write_json, write_jsonl
from .semantic_data import normalized, validate_semantics

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

COLOR_WORDS = ("light colored", "black", "white", "red", "blue", "yellow", "green",
               "brown", "pink", "purple", "orange", "blonde", "blond", "gray", "grey",
               "tan", "silver", "gold")
SIZE_WORDS = ("little", "small", "tiny", "large", "big", "huge", "giant", "young",
              "old", "baby", "adult", "tall", "short", "long", "narrow", "wide")
LEXICON = [("color", word) for word in COLOR_WORDS] + [("size", word) for word in SIZE_WORDS]

STOPWORDS = {
    "a", "an", "the", "and", "with", "in", "on", "of", "to", "at", "by", "near",
    "next", "into", "onto", "or", "his", "her", "its", "their", "some", "few",
    "many", "two", "three", "one", "each", "every", "another", "is", "are", "was",
    "were", "has", "have", "being", "be", "beside", "behind", "front", "under",
    "over", "across", "along", "around", "down", "up", "while", "from", "for",
}
VERBISH = {
    "wearing", "covered", "filled", "performing", "doing", "standing", "sitting",
    "walking", "running", "playing", "looking", "holding", "carrying", "climbing",
    "chasing", "jumping", "spinning", "waving", "grinding", "riding", "attempting",
    "descending", "talking", "wears", "wore", "wear", "sits", "runs", "plays",
    "holds", "climbs", "chases", "jumps", "spins", "waves", "eats", "lays", "lies",
    "rides", "walks", "stand", "stands",
}
RELATION_ARG_PREFIXES = {
    "with", "in", "on", "at", "near", "beside", "by", "of", "to", "and", "or",
    "while", "under", "over", "behind", "next", "against", "from", "for",
}
EXCLUDED_OBJECT_PREFIXES = {"with", "of", "from", "by", "and", "or", "while", "beside", "near", "behind"}
AUX = {"is", "are", "was", "were", "has", "have", "had", "been", "being", "will",
       "would", "can", "could", "may", "might", "shall", "should", "does", "do", "did", "be"}
VERBISH_FINITE = {word for word in VERBISH if not word.endswith(("ing", "ed"))}

ADDITION_PAIRS = (
    (" The sky is blue.", " The sky is green."),
    (" It is daytime.", " It is nighttime."),
    (" There is one person in the picture.", " There are two people in the picture."),
    (" The photo was taken outdoors.", " The photo was taken indoors."),
)

LEXICON_LOW = {word.casefold() for _, word in LEXICON}
DRAFT_REVIEW = {"status": "draft", "disputes": []}


def _find_all(text, needle):
    index = text.find(needle)
    while index != -1:
        yield index
        index = text.find(needle, index + 1)


def _word_occurrences(text, word):
    pattern = re.compile(r"(?<![A-Za-z0-9-])" + re.escape(word) + r"(?![A-Za-z0-9-])", re.IGNORECASE)
    return len(pattern.findall(text))


def _contains_word(head, word):
    return _word_occurrences(head, word) > 0


def _phrase_verbish(phrase):
    words = phrase.split()
    return any(word.casefold() in VERBISH or word.endswith(("ing", "ed")) for word in words)


def _extract_verb_span(caption, gap_start, gap_end):
    """Deterministic verb-phrase recovery inside a verbatim gap of the caption.

    A token qualifies when it is a finite verb form, an auxiliary, or an
    -ing/-ed form attached to an auxiliary (or opening the gap). Returns the
    span of the first qualifying run containing a real verb, else None.
    """
    gap = caption[gap_start:gap_end]
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", gap)]
    runs, current = [], []
    for index, (token, start, end) in enumerate(tokens):
        low = token.casefold()
        attached = token.endswith(("ing", "ed")) and (index == 0 or tokens[index - 1][0].casefold() in AUX)
        if low in AUX or low in VERBISH_FINITE or attached:
            current.append((token, start, end))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    for run in runs:
        if any(token.casefold() not in AUX for token, _, _ in run):
            return (gap_start + run[0][1], gap_start + run[-1][2])
    return None


def _loose_normalized(text):
    return re.sub(r"\s*([.,!?;])\s*", r"\1", " ".join(text.casefold().split()))


def anchor(phrase, source):
    """Anchor a proposed phrase to a unique exact substring of the source.

    Underscore-joined phrases are split first. Matching is case-insensitive but
    the returned span always recovers the original bytes. Requires whole-word
    boundaries; returns (start, end) or None when missing or ambiguous.
    """
    phrase = " ".join(phrase.strip().replace("_", " ").split())
    if len(phrase) < 2 or phrase.strip(" .,;:!?") == "":
        return None
    boundary = r"(?<![A-Za-z0-9-])" + re.escape(phrase.casefold()) + r"(?![A-Za-z0-9-])"
    hits = [match.start() for match in re.finditer(boundary, source.casefold())]
    if len(hits) != 1:
        return None
    start = hits[0]
    actual = source[start:start + len(phrase)]
    if len(list(_find_all(source, actual))) != 1:
        return None
    return (start, start + len(phrase))


def _find_entity(entities, phrase):
    phrase_norm = normalized(phrase)
    for entity in entities:
        if normalized(entity["head"]) == phrase_norm:
            return entity
    for entity in entities:
        if _contains_word(entity["head"], phrase_norm.split(" ")[-1]):
            return entity
    return None


def parse_lines(output, caption):
    """Anchor LM parse lines; unanchorable proposals are excluded with reasons."""
    entities, attributes, relations, rejects = [], [], [], []
    seen_entities = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        prefix, rest = line.split(":", 1)
        kind = prefix.strip().upper()
        rest = rest.strip()
        if kind == "ENTITY":
            hit = anchor(rest, caption)
            if hit is None or len(rest) > 0.8 * len(caption):
                rejects.append({"stage": "parse", "item": raw, "reason": "entity phrase not uniquely anchored or implausibly long"})
                continue
            head = caption[hit[0]:hit[1]]
            key = (tuple(hit), normalized(head))
            if key in seen_entities:
                continue
            seen_entities.add(key)
            entities.append({"head": head, "span": list(hit), "attributes": []})
        elif kind == "ATTRIBUTE":
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                rejects.append({"stage": "parse", "item": raw, "reason": "attribute line must have owner | name | value"})
                continue
            owner = parts[0].split(":", 1)[1].strip()
            name = re.sub(r"\s+", "_", parts[1].replace("_", " ").strip())
            hit = anchor(parts[2], caption)
            if not name or hit is None:
                rejects.append({"stage": "parse", "item": raw, "reason": "attribute name or value not anchorable"})
                continue
            attributes.append((owner, {"name": name, "value": caption[hit[0]:hit[1]], "span": list(hit)}))
        elif kind == "RELATION":
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                rejects.append({"stage": "parse", "item": raw, "reason": "relation line must have subject | predicate | object"})
                continue
            subject = parts[0].split(":", 1)[1].strip()
            middle = parts[1:-1]
            if subject.split()[0].casefold() in RELATION_ARG_PREFIXES:
                rejects.append({"stage": "parse", "item": raw, "reason": "relation argument starts with a function word"})
                continue
            # The small model fuses several relations into one line; salvage
            # anchorable noun phrases from the middle as plain entities.
            for phrase in middle[1:]:
                if (phrase.split()[0].casefold() in RELATION_ARG_PREFIXES or _phrase_verbish(phrase)
                        or len(phrase) > 0.8 * len(caption)):
                    continue
                hit = anchor(phrase, caption)
                if hit is None:
                    continue
                head = caption[hit[0]:hit[1]]
                key = (tuple(hit), normalized(head))
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                entities.append({"head": head, "span": list(hit), "attributes": []})
            # Object: the first non-verbish middle phrase when the line was
            # fused, otherwise the last part.
            object_ = None
            for phrase in [*middle[1:], parts[-1]]:
                if phrase.split()[0].casefold() in RELATION_ARG_PREFIXES or _phrase_verbish(phrase):
                    continue
                if anchor(phrase, caption) is not None:
                    object_ = phrase
                    break
            if object_ is None:
                rejects.append({"stage": "parse", "item": raw, "reason": "no anchorable object phrase in relation line"})
                continue
            if subject.split()[0].casefold() in RELATION_ARG_PREFIXES:
                rejects.append({"stage": "parse", "item": raw, "reason": "relation argument starts with a function word"})
                continue
            if normalized(subject) == normalized(object_):
                rejects.append({"stage": "parse", "item": raw, "reason": "subject and object are the same phrase"})
                continue
            subject_hit = anchor(subject, caption)
            if subject_hit is None and subject.split()[0].casefold() in RELATION_ARG_PREFIXES:
                subject_hit = anchor(" ".join(subject.split()[1:]), caption)
            object_hit = anchor(object_, caption)
            if object_hit is None and object_.split()[0].casefold() in RELATION_ARG_PREFIXES:
                object_hit = anchor(" ".join(object_.split()[1:]), caption)
            if subject_hit is None or object_hit is None:
                rejects.append({"stage": "parse", "item": raw, "reason": "relation argument not uniquely anchored"})
                continue
            if subject_hit[0] >= object_hit[0]:
                rejects.append({"stage": "parse", "item": raw, "reason": "relation arguments not in left-to-right order"})
                continue
            predicate = middle[0]
            predicate_hit = anchor(predicate, caption)
            if predicate_hit is None:
                predicate_hit = _extract_verb_span(caption, subject_hit[1], object_hit[0])
                if predicate_hit is None:
                    rejects.append({"stage": "parse", "item": raw,
                                    "reason": f"no verbatim or derivable predicate: {predicate!r}"})
                    continue
            relations.append((caption[subject_hit[0]:subject_hit[1]], subject_hit,
                              predicate_hit, caption[object_hit[0]:object_hit[1]], object_hit))
    return entities, attributes, relations, rejects


def _next_noun(caption, position):
    for match in re.finditer(r"[A-Za-z]+", caption[position:]):
        token = match.group(0)
        low = token.casefold()
        if low in STOPWORDS or low in LEXICON_LOW:
            continue
        return token, position + match.start()
    return None, None


def _lexicon_pattern():
    words = sorted({word for _, word in LEXICON}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(word) for word in words) + r")(?![A-Za-z0-9-])", re.IGNORECASE)


def lexicon_attributes(caption, entities):
    """Deterministic color/size supplement; owners resolved from anchored entities."""
    added, rejects = [], []
    classes = {word.casefold(): kind for kind, word in LEXICON}
    for match in _lexicon_pattern().finditer(caption):
        value = match.group(0)
        start, end = match.span()
        noun, noun_pos = _next_noun(caption, end)
        if noun is None or noun.casefold() in VERBISH:
            continue
        owner = next((e for e in entities if e["span"][0] == noun_pos), None)
        if owner is None:
            owner = next((e for e in entities if e["span"][0] <= start and _contains_word(e["head"], noun)), None)
        if owner is None:
            preceding = [e for e in entities if e["span"][1] <= start and start - e["span"][1] <= 20]
            owner = max(preceding, key=lambda e: e["span"][1]) if preceding else None
        if owner is None:
            rejects.append({"stage": "attributes", "item": value, "reason": "no anchored entity owns this value"})
            continue
        name = f"{noun}_{classes[value.casefold()]}"
        if any(normalized(a["value"]) == normalized(value) for a in owner["attributes"]):
            continue
        owner["attributes"].append({"name": name, "value": value, "span": [start, end]})
        added.append((owner["head"], name, value))
    return added, rejects


def build_variants(caption, entities, relations, paraphrase, rng):
    """Deterministic templates over anchored parses; every foil is one edit."""
    variants, units, rejects = [], [], []
    by_id = {entity["id"]: entity for entity in entities}

    attribute_items = [(entity, attr) for entity in entities for attr in entity["attributes"]]
    pairs = []
    for i in range(len(attribute_items)):
        for j in range(i + 1, len(attribute_items)):
            (e1, a1), (e2, a2) = attribute_items[i], attribute_items[j]
            if normalized(a1["value"]) == normalized(a2["value"]):
                continue
            (s1, f1), (s2, f2) = a1["span"], a2["span"]
            if s1 <= s2 < f1 or s2 <= s1 < f2:
                continue
            pairs.append((e1, a1, e2, a2))
    if pairs:
        e1, a1, e2, a2 = rng.choice(pairs)
        lo = min(a1["span"][0], a2["span"][0])
        hi = max(a1["span"][1], a2["span"][1])
        before = caption[lo:hi]
        if a1["span"][0] < a2["span"][0]:
            (v_first, (fs, fe)), (v_second, (ss, se)) = (a1["value"], a1["span"]), (a2["value"], a2["span"])
        else:
            (v_first, (fs, fe)), (v_second, (ss, se)) = (a2["value"], a2["span"]), (a1["value"], a1["span"])
        after = before[:fs - lo] + v_second + before[fe - lo:ss - lo] + v_first + before[se - lo:]
        if after == before:
            rejects.append({"stage": "binding", "item": "attribute swap", "reason": "swap left text unchanged"})
        else:
            entity_ids = sorted({e1["id"], e2["id"]})
            variants.append({"id": "v-binding", "type": "minimal_edit", "text": caption[:lo] + after + caption[hi:],
                             "target_unit_id": e1["id"], "phenomenon": "attribute_binding",
                             "semantic_relation": "changed", "visual_label": "uncertain", "naturalness": "uncertain",
                             "edit": {"field": f"{e1['id']},{e2['id']}.attribute_binding", "span": [lo, hi],
                                      "before": before, "after": after}, "review": dict(DRAFT_REVIEW)})
            units.append({"id": "u-binding", "type": "entity", "entity_ids": entity_ids, "span": [lo, hi],
                          "text": before, "foil": after, "variant_id": "v-binding", "review": dict(DRAFT_REVIEW)})
    else:
        rejects.append({"stage": "binding", "item": "attribute swap", "reason": "no two distinct attribute values to swap"})

    direction_candidates = []
    for relation in relations:
        subject, object_ = by_id[relation["subject"]], by_id[relation["object"]]
        p0, p1 = relation["span"]
        subject_np = caption[:p0].strip()
        tail = caption[p1:]
        punctuation = re.search(r"[.,!?;]", tail)
        if punctuation:
            object_np = tail[:punctuation.start()].strip()
            end = p1 + punctuation.start()
        else:
            object_np = tail.strip()
            end = len(caption)
        predicate = caption[p0:p1]
        finite_verb = (" " not in predicate and not predicate.endswith(("ing", "ed"))
                       and predicate.casefold() not in AUX)
        if (not subject_np or not object_np or not finite_verb
                or any(char in ".,!?;" for char in subject_np + object_np)
                or object_np.split()[0].casefold() in EXCLUDED_OBJECT_PREFIXES
                or normalized(subject_np) == normalized(object_np)
                or _word_occurrences(subject_np, subject["head"]) != 1
                or _word_occurrences(object_np, object_["head"]) != 1
                or _contains_word(object_np, subject["head"]) or _contains_word(subject_np, object_["head"])):
            continue
        direction_candidates.append((relation, subject, object_, subject_np, object_np, end))
    if direction_candidates:
        relation, subject, object_, subject_np, object_np, end = rng.choice(direction_candidates)
        predicate = caption[relation["span"][0]:relation["span"][1]]
        after = object_np[:1].upper() + object_np[1:] + " " + predicate + " " + subject_np[:1].lower() + subject_np[1:]
        variants.append({"id": "v-direction", "type": "minimal_edit", "text": after + caption[end:],
                         "target_unit_id": relation["id"], "phenomenon": "relation_direction",
                         "semantic_relation": "changed", "visual_label": "uncertain", "naturalness": "uncertain",
                         "edit": {"field": f"{relation['id']}.direction", "span": [0, end],
                                  "before": caption[:end], "after": after}, "review": dict(DRAFT_REVIEW)})
        units.append({"id": "u-direction", "type": "relation", "entity_ids": [subject["id"], object_["id"]],
                      "span": [0, len(caption)], "text": caption, "foil": after + caption[end:],
                      "variant_id": "v-direction", "review": dict(DRAFT_REVIEW)})
    else:
        rejects.append({"stage": "direction", "item": "relation direction swap", "reason": "no simple subject-verb-object layout to reverse"})

    clean = " ".join(paraphrase.split()).strip().strip('`"\'')
    if not clean:
        rejects.append({"stage": "paraphrase", "item": paraphrase, "reason": "empty paraphrase proposal"})
    elif _loose_normalized(clean) == _loose_normalized(caption):
        rejects.append({"stage": "paraphrase", "item": clean, "reason": "identical to the original caption"})
    elif len(clean) > 2.5 * len(caption) or len(clean) < 0.5 * len(caption):
        rejects.append({"stage": "paraphrase", "item": clean, "reason": "implausible paraphrase length"})
    else:
        variants.append({"id": "v-paraphrase", "type": "paraphrase", "text": clean, "target_unit_id": "sentence",
                         "phenomenon": "paraphrase", "semantic_relation": "equivalent",
                         "visual_label": "uncertain", "naturalness": "uncertain",
                         "edit": {"field": "sentence", "span": [0, len(caption)], "before": caption, "after": clean},
                         "review": dict(DRAFT_REVIEW)})

    true_add, false_add = rng.choice(ADDITION_PAIRS)
    for kind, detail, variant_id in (("true_addition", true_add, "v-add-true"), ("false_addition", false_add, "v-add-false")):
        variants.append({"id": variant_id, "type": kind, "text": caption + detail, "target_unit_id": "sentence",
                         "phenomenon": "false_addition", "semantic_relation": "changed",
                         "visual_label": "uncertain", "naturalness": "uncertain", "control_pair_id": "addition-pair",
                         "edit": {"field": "sentence", "span": [len(caption), len(caption)], "before": "", "after": detail},
                         "review": dict(DRAFT_REVIEW)})
    return variants, units, rejects


def generate_record(row, backend, prompts, prompts_sha256, seed, code_revision):
    caption = row["text"]
    rng = random.Random(f"{seed}:{row['sample_id']}")
    parse_output = backend.generate(_messages(prompts["parse"], caption), prompts["parse"]["max_new_tokens"], seed)
    entities, attributes, relations, rejects = parse_lines(parse_output, caption)
    for index, entity in enumerate(entities):
        entity["id"] = f"e{index + 1}"
    for owner, attr in attributes:
        matched = False
        for entity in entities:
            if normalized(entity["head"]) != normalized(owner):
                continue
            matched = True
            if any(normalized(a["value"]) == normalized(attr["value"]) for a in entity["attributes"]):
                break  # lexicon/LM overlap: the same value is already recorded
            attr["entity_id"] = entity["id"]
            entity["attributes"].append(attr)
            break
        if not matched:
            rejects.append({"stage": "parse", "item": f"{owner}: {attr['name']}={attr['value']}",
                            "reason": "attribute owner phrase does not match any anchored entity"})
    added_attributes, attribute_rejects = lexicon_attributes(caption, entities)
    rejects.extend(attribute_rejects)
    for entity in entities:
        for attr in entity["attributes"]:
            if attr.get("entity_id") is None:
                attr["entity_id"] = entity["id"]
    final_relations = []
    seen_relation_keys = set()
    for subject, subject_hit, predicate_hit, object_, object_hit in relations:
        subject_entity = _find_entity(entities, subject)
        if subject_entity is None:
            subject_entity = {"id": f"e{len(entities) + 1}", "head": subject, "span": list(subject_hit), "attributes": []}
            entities.append(subject_entity)
        object_entity = _find_entity(entities, object_)
        if object_entity is None:
            object_entity = {"id": f"e{len(entities) + 1}", "head": object_, "span": list(object_hit), "attributes": []}
            entities.append(object_entity)
        if subject_entity["id"] == object_entity["id"]:
            rejects.append({"stage": "parse", "item": f"{subject} -> {object_}", "reason": "relation subject and object resolve to one entity"})
            continue
        key = (subject_entity["id"], normalized(caption[predicate_hit[0]:predicate_hit[1]]), object_entity["id"])
        if key in seen_relation_keys:
            continue
        seen_relation_keys.add(key)
        final_relations.append({"id": f"r{len(final_relations) + 1}", "subject": subject_entity["id"],
                                "object": object_entity["id"], "predicate": caption[predicate_hit[0]:predicate_hit[1]],
                                "span": list(predicate_hit)})
    paraphrase = backend.generate(_messages(prompts["paraphrase"], caption), prompts["paraphrase"]["max_new_tokens"], seed)
    variants, units, variant_rejects = build_variants(caption, entities, final_relations, paraphrase, rng)
    rejects.extend(variant_rejects)

    notes = [f"{item['stage']}: {item['item']!r} excluded ({item['reason']})" for item in rejects]
    notes.append(f"lexicon attributes added: {len(added_attributes)}")
    record = dict(row)
    record["entities"] = entities
    record["relations"] = final_relations
    record["variants"] = variants
    record["units"] = units
    record["provenance"] = {"source_manifest_sha256": row["provenance"]["source_manifest_sha256"],
                            "generator": {**backend.provenance(), "prompt_sha256": prompts_sha256},
                            "seed": seed, "code_revision": code_revision}
    record["generation_notes"] = notes
    return record, rejects, {"parse_output": parse_output, "paraphrase_output": paraphrase}


def _messages(prompt_config, caption):
    messages = [{"role": "system", "content": prompt_config["system"]}]
    for user, assistant in prompt_config["examples"]:
        messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
    messages.append({"role": "user", "content": caption})
    return messages


def generate_semantics(samples, bundle, data_root, output, backend, prompts_path, seed=42,
                       code_revision="working-tree", limit=None):
    if type(seed) is not int or (limit is not None and (type(limit) is not int or limit <= 0)):
        raise ValueError("seed must be an integer and limit a positive integer")
    prompts = read_json(prompts_path)
    if not isinstance(prompts, dict) or "parse" not in prompts or "paraphrase" not in prompts:
        raise ValueError("unsupported generation prompts file")
    prompts_sha256 = file_digest(prompts_path)
    rows = read_jsonl(samples)
    if limit is not None:
        rows = rows[:limit]
    results, rejections, raw_outputs = [], [], []
    for row in rows:
        record, rejects, raw = generate_record(row, backend, prompts, prompts_sha256, seed, code_revision)
        results.append(record)
        rejections.append({"sample_id": row["sample_id"], "items": rejects})
        raw_outputs.append({"sample_id": row["sample_id"], "parse_output": raw["parse_output"],
                            "paraphrase_output": raw["paraphrase_output"]})
    with new_directory(output) as target:
        write_jsonl(target / "samples.work.jsonl", results)
        write_jsonl(target / "rejections.jsonl", rejections)
        write_jsonl(target / "raw-outputs.jsonl", raw_outputs)
        write_json(target / "prompts.json", prompts)
        audit = validate_semantics(target / "samples.work.jsonl", bundle, "draft", data_root)
        summary = [{"sample_id": row["sample_id"], "entities": len(row["entities"]),
                    "attributes": sum(len(e["attributes"]) for e in row["entities"]),
                    "relations": len(row["relations"]), "variants": [v["id"] for v in row["variants"]],
                    "units": len(row["units"]), "rejections": len(items), "notes": row["generation_notes"]}
                   for row, items in zip(results, rejections)]
        write_json(target / "generation.json", {
            "schema_version": "fgir-semantic-generation-v1",
            "generator": {**backend.provenance(), "prompt_sha256": prompts_sha256},
            "backend": getattr(backend, "describe", lambda: {})(),
            "prompts_sha256": prompts_sha256, "seed": seed, "code_revision": code_revision,
            "lexicon": {"color": list(COLOR_WORDS), "size": list(SIZE_WORDS)},
            "addition_pairs": [list(pair) for pair in ADDITION_PAIRS],
            "determinism": "greedy decoding; per-record template choices seeded by random.Random(f'{seed}:{sample_id}')",
            "samples_sha256": file_digest(target / "samples.work.jsonl"),
            "input": {"samples": str(samples), "samples_sha256": file_digest(samples), "prompts": str(prompts_path)},
            "summary": summary,
            "limitation": ("Candidates are drafts. Visual truth, semantic equivalence and object binding need "
                           "human review; structural validation does not certify them.")})
        write_json(target / "receipt.json", {"samples_sha256": file_digest(target / "samples.work.jsonl"),
                                             "prompts_sha256": prompts_sha256, "audit": audit})
    return {"output": str(output), "records": len(results),
            "entities": sum(len(r["entities"]) for r in results),
            "relations": sum(len(r["relations"]) for r in results),
            "variants": sum(len(r["variants"]) for r in results),
            "units": sum(len(r["units"]) for r in results),
            "rejected_items": sum(len(r["items"]) for r in rejections),
            "samples_sha256": file_digest(Path(output) / "samples.work.jsonl"),
            "prompts_sha256": prompts_sha256, "backend": backend.provenance()["name"], **audit}


class GenerationBackend:
    """Proposes raw text only; never certifies review status."""

    def __init__(self, name, weights_sha256):
        if not isinstance(weights_sha256, str) or len(weights_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in weights_sha256):
            raise ValueError("weights_sha256 must be a lowercase SHA-256 of the actual weight files")
        self.name = name
        self.weights_sha256 = weights_sha256

    def generate(self, messages, max_new_tokens, seed):
        raise NotImplementedError

    def provenance(self):
        return {"name": self.name, "weights_sha256": self.weights_sha256, "prompt_sha256": None}

    def describe(self):
        return {}


class LocalLmBackend(GenerationBackend):
    """Lazy transformers backend; weights digest computed from the real files."""

    def __init__(self, model_id=DEFAULT_MODEL_ID, allow_download=True):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._environment = {}
        snapshot = self._resolve_snapshot(allow_download)
        self.snapshot_dir = str(snapshot)
        weight_files = sorted(str(p) for p in Path(snapshot).rglob("*.safetensors"))
        if not weight_files:
            raise ValueError(f"no .safetensors weights found under {self.snapshot_dir}")
        digest = hashlib.sha256()
        for path in weight_files:
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        super().__init__(f"local-{model_id.split('/')[-1].lower()}-templates-v1", digest.hexdigest())
        self._weight_files = [str(Path(p).name) for p in weight_files]

    def _resolve_snapshot(self, allow_download):
        home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        model_dir = home / "hub" / f"models--{self.model_id.replace('/', '--')}"
        if model_dir.exists():
            for snapshot in sorted((model_dir / "snapshots").glob("*")):
                if any(snapshot.rglob("*.safetensors")):
                    return snapshot
        if not allow_download:
            raise ValueError(
                f"local generation weights unavailable for {self.model_id!r}: no cached snapshot and "
                "downloads disabled. Next steps: obtain the weights locally (this machine needs "
                "HF_ENDPOINT=https://hf-mirror.com and HF_HUB_DISABLE_XET=1), e.g. python -c "
                f"\"from huggingface_hub import snapshot_download; snapshot_download('{self.model_id}')\", "
                "then re-run. No model call and no fabricated weight summary was produced.")
        try:
            from huggingface_hub import snapshot_download
            return Path(snapshot_download(repo_id=self.model_id))
        except Exception as exc:  # network or registry failure: report, do not fake anything
            raise ValueError(
                f"local generation weights unavailable for {self.model_id!r}: no cached snapshot and "
                f"download failed ({exc}). Next steps: obtain the weights locally (this machine needs "
                "HF_ENDPOINT=https://hf-mirror.com and HF_HUB_DISABLE_XET=1), then re-run. "
                "No model call and no fabricated weight summary was produced.") from exc

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            import transformers
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.snapshot_dir)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(self.snapshot_dir)
            self._model.eval()
            self._environment = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu"}

    def generate(self, messages, max_new_tokens, seed):
        self._ensure_loaded()
        import torch
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
                                          pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id)
        return self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def describe(self):
        return {"model_id": self.model_id, "snapshot_dir": self.snapshot_dir,
                "weight_files": self._weight_files, "environment": self._environment}
