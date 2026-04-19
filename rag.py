"""
Lightweight RAG system for solar grid management guidelines.

Uses FAISS (in-memory) + HuggingFace sentence-transformers to retrieve
the most relevant operational guidelines given a natural-language query.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Grid management knowledge base ───────────────────────────────────────────
GRID_GUIDELINES = [
    "Store excess solar energy in batteries during peak generation hours "
    "(10 AM – 2 PM) to maximize self-consumption and reduce grid export.",

    "Use battery backup systems during low generation periods to maintain "
    "uninterrupted power supply and grid stability.",

    "Balance grid load during generation fluctuations by activating "
    "demand-response protocols across connected loads.",

    "Curtail non-essential loads when solar generation drops below 20% of "
    "rated capacity to prevent grid voltage instability.",

    "Schedule high-energy industrial processes during peak solar generation "
    "windows to take advantage of free renewable energy.",

    "Monitor inverter efficiency and reduce output if module temperature "
    "exceeds safe thresholds to prevent equipment degradation.",

    "Activate grid feed-in during surplus generation periods to offset "
    "utility costs and contribute to grid decarbonization.",

    "Pre-charge battery storage before anticipated low-irradiation periods "
    "such as cloudy or rainy forecasts to ensure backup availability.",

    "Implement load-shedding protocols when generation variability exceeds "
    "safe operating margins to protect sensitive equipment.",

    "Perform preventive maintenance during consistently low-generation "
    "periods to minimize production downtime impact.",

    "Diversify energy sources during extended low-irradiation seasons to "
    "ensure continuous supply and reduce dependency on solar alone.",

    "Use real-time monitoring dashboards to detect and respond to sudden "
    "generation drops immediately, enabling rapid grid rebalancing.",

    "Reduce reliance on grid import during high-tariff hours by discharging "
    "stored solar energy from battery systems.",

    "Stagger startup of heavy machinery across time slots to avoid demand "
    "spikes that exceed available solar generation capacity.",
]

# ── Lazy-loaded singleton objects ─────────────────────────────────────────────
_model = None   # type: SentenceTransformer | None
_index = None   # type: faiss.IndexFlatIP | None


def _build_index():
    """Build the FAISS index from guideline embeddings (called once)."""
    global _model, _index

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = _model.encode(
        GRID_GUIDELINES,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Inner-product on L2-normalized vectors == cosine similarity
    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)


def retrieve_guidelines(query, top_k=3):
    """Retrieve the most relevant grid management guidelines for a query.

    Parameters
    ----------
    query : str
        Natural-language description of the current grid situation,
        e.g. "high variability in solar output with sudden drops".
    top_k : int
        Number of guidelines to return (default 3).

    Returns
    -------
    list[dict]
        Each dict has:
          - 'guideline' (str): the matched guideline text
          - 'score' (float):   cosine-similarity score (0 – 1)
    """
    if _index is None:
        _build_index()

    q_emb = _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = _index.search(q_emb, top_k)

    return [
        {
            "guideline": GRID_GUIDELINES[idx],
            "score":     round(float(sc), 4),
        }
        for sc, idx in zip(scores[0], indices[0])
    ]
