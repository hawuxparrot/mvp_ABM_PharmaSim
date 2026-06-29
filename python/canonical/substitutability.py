"""ATC-based product substitutability.

The WHO Anatomical Therapeutic Chemical (ATC) code is a 7-character hierarchical key:

    position  example  level  meaning
    1         C        1      anatomical main group
    2-3       10       2      therapeutic subgroup
    4         A        3      pharmacological subgroup
    5         A        4      chemical subgroup
    6-7       05       5      chemical substance (active ingredient)

Two products with the same ATC5 share the active substance (e.g. Atorvastatin 20mg vs
40mg); the same ATC4 means the same chemical class (e.g. atorvastatin vs simvastatin, both
statins). We model interchangeability by walking the prefix tree up to a configurable
cutoff level: smaller level == looser substitution.

Caveat: ATC says nothing about dose, form, or bioequivalence, so ATC-level substitution only
approximates clinical interchangeability. A manual ``substitution_group_id`` override lets a
modeller force (or forbid) groupings the ATC hierarchy gets wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from policy.models import Product

# (start, end) slice bounds of each ATC level within the 7-char code.
_LEVEL_BOUNDS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (0, 3),
    3: (0, 4),
    4: (0, 5),
    5: (0, 7),
}
_MIN_LEVEL = 1
_MAX_LEVEL = 5


def atc_prefix(code: str, level: int) -> str:
    """Return the ATC prefix for ``code`` truncated to ``level`` (1..5)."""
    if level < _MIN_LEVEL or level > _MAX_LEVEL:
        raise ValueError(f"ATC level must be in [{_MIN_LEVEL}, {_MAX_LEVEL}], got {level}")
    start, end = _LEVEL_BOUNDS[level]
    return (code or "").strip().upper()[start:end]


def _segments(code: str) -> list[str]:
    """Split an ATC code into its per-level segments for trie insertion."""
    code = (code or "").strip().upper()
    return [
        code[0:1],
        code[1:3],
        code[3:4],
        code[4:5],
        code[5:7],
    ]


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    product_ids: list[str] = field(default_factory=list)  # products terminating at this node


class SubstitutabilityIndex:
    """Prefix tree over ATC codes answering "what can substitute for product X?".

    Parameters
    ----------
    products:
        Catalog products. Products without a (valid) ATC code participate only via manual
        groups.
    default_level:
        Cutoff used when ``substitutes`` is called without an explicit level. 5 = same active
        substance, 4 = same chemical subgroup (typical clinical substitution), etc.
    group_overrides:
        ``product_id -> group_id``. Products sharing a group_id are always mutual substitutes,
        independent of (and unioned with) the ATC result.
    """

    def __init__(
        self,
        products: list[Product],
        *,
        default_level: int = 5,
        group_overrides: dict[str, str] | None = None,
    ) -> None:
        if default_level < _MIN_LEVEL or default_level > _MAX_LEVEL:
            raise ValueError("default_level must be in [1, 5]")
        self.default_level = default_level
        self._root = _TrieNode()
        self._atc_by_product: dict[str, str] = {}
        self._group_by_product: dict[str, str] = dict(group_overrides or {})
        self._group_members: dict[str, list[str]] = {}

        for product in products:
            self._insert(product)
        for pid, gid in self._group_by_product.items():
            self._group_members.setdefault(gid, [])
            if pid not in self._group_members[gid]:
                self._group_members[gid].append(pid)

    def _insert(self, product: Product) -> None:
        code = (product.atc_code or "").strip().upper()
        if not code:
            return
        self._atc_by_product[product.ext_id] = code
        node = self._root
        for seg in _segments(code):
            if not seg:
                break
            node = node.children.setdefault(seg, _TrieNode())
        node.product_ids.append(product.ext_id)

    def _node_at_prefix(self, code: str, level: int) -> _TrieNode | None:
        node = self._root
        for seg in _segments(code)[:level]:
            if not seg:
                return None
            child = node.children.get(seg)
            if child is None:
                return None
            node = child
        return node

    @staticmethod
    def _collect(node: _TrieNode, out: list[str]) -> None:
        out.extend(node.product_ids)
        for child in node.children.values():
            SubstitutabilityIndex._collect(child, out)

    def substitutes(
        self,
        product_id: str,
        *,
        level: int | None = None,
        include_self: bool = False,
    ) -> list[str]:
        """Products interchangeable with ``product_id`` at the given ATC ``level``.

        The result unions ATC-prefix matches with any manual-group members. Order is stable
        (insertion order); ``include_self`` controls whether ``product_id`` itself appears.
        """
        lvl = self.default_level if level is None else level
        result: list[str] = []
        seen: set[str] = set()

        def add(pid: str) -> None:
            if pid in seen:
                return
            if not include_self and pid == product_id:
                return
            seen.add(pid)
            result.append(pid)

        code = self._atc_by_product.get(product_id)
        if code is not None:
            node = self._node_at_prefix(code, lvl)
            if node is not None:
                collected: list[str] = []
                self._collect(node, collected)
                for pid in collected:
                    add(pid)

        gid = self._group_by_product.get(product_id)
        if gid is not None:
            for pid in self._group_members.get(gid, []):
                add(pid)

        return result

    def are_substitutes(self, a: str, b: str, *, level: int | None = None) -> bool:
        if a == b:
            return True
        return b in self.substitutes(a, level=level, include_self=False)
