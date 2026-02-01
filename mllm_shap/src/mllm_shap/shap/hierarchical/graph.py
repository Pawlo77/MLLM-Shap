"""Computation graph for hierarchical SHAP values."""

from torch import Tensor
from pydantic import BaseModel, ConfigDict


class GraphNode(BaseModel):
    """Node in the hierarchical explanation computation graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shap_values: Tensor | None = None
    """SHAP values for the group represented by this node."""

    children: list["GraphNode"] | None = None
    """ Child nodes representing subgroups. """

    group_ids: Tensor | None = None
    """Group IDs tensor if applicable."""

    group_mask: Tensor | None = None
    """Group mask tensor if applicable."""

    def display(self) -> None:
        """Display the graph node and its children."""
        # TODO
        print(self)
