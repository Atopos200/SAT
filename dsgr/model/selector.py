"""Bridge module for DSGR selector.

Current implementation lives in `innovation.subgraph_selector`.
This bridge keeps compatibility while enabling staged refactoring.
"""

from innovation.subgraph_selector import (  # noqa: F401
    AdaptiveSubgraphSelector,
    KGIndex,
    NeighborScorer,
    SelectedSubgraph,
)

