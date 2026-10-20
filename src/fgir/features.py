"""Normalized exact cosine index over cached features with adapter rebuild hooks."""

from pathlib import Path

from .data import file_digest
from .encoding import load_features
from .files import read_json, write_json


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise ValueError("index building requires: python -m pip install -e '.[model]'") from exc
    return numpy


def normalize_rows(matrix):
    numpy = _numpy()
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero embeddings cannot be normalized")
    return matrix / norms


class FeatureIndex:
    """Vectors must share one model and preprocessing version; ids stay strings."""

    def __init__(self, ids, vectors, meta=None):
        numpy = _numpy()
        if len(ids) != vectors.shape[0] or not ids:
            raise ValueError("index ids and vectors differ")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate image_id in index")
        if not numpy.isfinite(vectors).all():
            raise ValueError("index contains non-finite values")
        self.ids = list(ids)
        self.matrix = normalize_rows(vectors.astype(numpy.float32))
        self.dimension = self.matrix.shape[1]
        self.meta = dict(meta or {})
        self._id_rank = numpy.empty(len(self.ids), dtype=numpy.int64)
        self._id_rank[numpy.argsort(numpy.array(self.ids, dtype="S"), kind="stable")] = numpy.arange(len(self.ids))

    @classmethod
    def from_features(cls, path, adapter=None):
        """Build from cached raw features; adapter maps raw rows before normalization."""
        ids, vectors, meta = load_features(path)
        if adapter is not None:
            vectors = adapter(vectors)
        return cls(ids, vectors, meta)

    def search(self, query_vector, top_k=10):
        return self.search_many([query_vector], top_k)[0]

    def search_many(self, query_vectors, top_k=10):
        numpy = _numpy()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        queries = numpy.asarray(query_vectors, dtype=numpy.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError("query dimension does not match index")
        queries = normalize_rows(queries)
        scores = queries @ self.matrix.T
        id_array = numpy.array(self.ids)
        results = []
        for row in scores:
            order = numpy.lexsort((self._id_rank, -row))
            top = order[:top_k]
            results.append([{"image_id": str(id_array[i]), "score": float(row[i])} for i in top])
        return results

    def save(self, path):
        numpy = _numpy()
        path = Path(path)
        numpy.savez(path, ids=numpy.array(self.ids), vectors=self.matrix)
        write_json(Path(str(path) + ".meta.json"), {**self.meta, "count": len(self.ids),
                                                    "dimension": self.dimension,
                                                    "normalization": "l2_rows",
                                                    "index_sha256": file_digest(path)})

    @classmethod
    def load(cls, path):
        numpy = _numpy()
        path = Path(path)
        meta = read_json(Path(str(path) + ".meta.json"))
        if meta.get("index_sha256") != file_digest(path):
            raise ValueError(f"index hash mismatch: {path.name}")
        with numpy.load(path, allow_pickle=False) as data:
            ids = [str(value) for value in data["ids"]]
            vectors = data["vectors"].astype(numpy.float32)
        return cls(ids, vectors, meta)


def build_index(features_path, output, adapter=None):
    """Build and publish a normalized index; C's adapter maps raw rows before L2."""
    index = FeatureIndex.from_features(features_path, adapter=adapter)
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    index.save(output)
    return {"output": str(output), "count": len(index.ids), "dimension": index.dimension,
            "normalization": "l2_rows", "model_id": index.meta.get("model_id"),
            "weights_sha256": index.meta.get("weights_sha256"),
            "note": "raw features stay cached; rebuilding through an adapter never re-encodes"}
