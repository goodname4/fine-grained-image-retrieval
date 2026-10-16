"""Small exact cosine index, independent of any AI service or encoder."""

from math import hypot, isfinite
from numbers import Real


def normalize(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding must be a non-empty numeric list")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in vector):
        raise ValueError("embedding values must be numbers, not booleans or strings")
    values = tuple(float(x) for x in vector)
    if not all(isfinite(x) for x in values):
        raise ValueError("embedding contains non-finite values")
    scale = max(abs(x) for x in values)
    if scale == 0:
        raise ValueError("zero embeddings cannot be normalized")
    scaled = tuple(x / scale for x in values)
    norm = hypot(*scaled)
    return tuple(x / norm for x in scaled)


class ExactIndex:
    """All vectors must come from compatible model and preprocessing versions."""

    def __init__(self, records):
        if not isinstance(records, list) or not records:
            raise ValueError("index must be a non-empty list")
        self.vectors = {}
        self.dimension = None
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("each index record must be an object")
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image_id must be a non-empty string")
            if image_id in self.vectors:
                raise ValueError(f"duplicate image_id: {image_id}")
            vector = normalize(row.get("embedding"))
            if self.dimension is None:
                self.dimension = len(vector)
            if len(vector) != self.dimension:
                raise ValueError("index embedding dimensions differ")
            self.vectors[image_id] = vector

    def search(self, query_vector, top_k=10):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query = normalize(query_vector)
        if len(query) != self.dimension:
            raise ValueError("query dimension does not match index")
        scores = [
            {"image_id": image_id, "score": sum(a * b for a, b in zip(query, vector))}
            for image_id, vector in self.vectors.items()
        ]
        # Stable deterministic tie-break, independent of input record order.
        scores.sort(key=lambda item: (-item["score"], item["image_id"]))
        return scores[:top_k]
