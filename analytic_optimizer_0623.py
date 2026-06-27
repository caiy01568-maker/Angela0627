# !/usr/bin/env python3
"""
Usage:
    from analytic_optimizer_0623 import AnalyticOptimizer

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.analytic_optimizer = AnalyticOptimizer(verbose=verbose)  ## REMEMBER TO ADD THIS LINE IN YOUR OPTIMIZER CLASS


    return self.analytic_optimizer.analytic_solver(
        block_count, 
        area_targets, 
        b2b_connectivity, 
        p2b_connectivity,
        pins_pos, 
        constraints, 
        target_positions
    )
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch

EPS = 1e-7
AREA_TOL = 0.01
Position = Tuple[float, float, float, float]

@dataclass
class BlockMeta: # Stores deterministic information for one block
    index: int
    area: float
    fixed: bool
    preplaced: bool
    mib: int
    cluster: int
    boundary: int
    width: float
    height: float
    target_x: Optional[float] = None
    target_y: Optional[float] = None

@dataclass
class AnalyticalConfig: # Hyperparameters for analytical placement.
    dtype: torch.dtype = torch.float64
    laplacian_reg: float = 1e-3
    preplaced_anchor: float = 1e6
    spread_iters: int = 12
    spread_step: float = 0.12
    spread_weight_start: float = 0.20
    spread_weight_growth: float = 0.05
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0

"""Generate analytical placement candidates before hard legalization."""
class AnalyticOptimizer():

    def __init__(self, verbose: bool = False):
        self.config = AnalyticalConfig()
        self.verbose = verbose

    """Return one analytical floorplan candidate."""
    def analytic_solver(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Position]:

        # build the meta data of each block
        meta = self._build_meta(
            block_count = block_count,
            area_targets = area_targets,
            constraints = constraints,
            target_positions = target_positions,
        )

        # solve the Laplacian system (analytical placement)
        center_x, center_y, degree = self._solve_laplacian_centers(
            meta=meta,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
        )

        # assign the shapes for MIB blocks (not violate the hard area constraints)
        self._assign_soft_shapes(
            meta=meta,
            center_x=center_x,
            center_y=center_y,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
        )

        self._apply_safe_mib_shapes(meta)

        positions = self._centers_to_rectangles(
            meta=meta,
            center_x=center_x,
            center_y=center_y,
        )

        positions = self._rough_spread(meta=meta, positions=positions)

        return positions

    """Convert a tensor-like object to Python rows."""
    @staticmethod
    def _rows(value):
        return value.detach().cpu().tolist() if hasattr(value, "detach") else list(value)
    
    """Build block metadata from raw contest tensors."""
    def _build_meta(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[BlockMeta]:

        meta: List[BlockMeta] = []

        areas = area_targets.detach().cpu().float().tolist()
        fixed_flags = constraints[:, 0].detach().cpu().bool().tolist()
        preplaced_flags = constraints[:, 1].detach().cpu().bool().tolist()
        mib_ids = constraints[:, 2].detach().cpu().int().tolist()
        cluster_ids = constraints[:, 3].detach().cpu().int().tolist()
        boundary_codes = constraints[:, 4].detach().cpu().int().tolist()

        target_rows = (
            target_positions.detach().cpu().float().tolist()
            if target_positions is not None
            else None
        )

        for i in range(block_count):
            area = float(areas[i])
            fixed = bool(fixed_flags[i])
            preplaced = bool(preplaced_flags[i])
            has_target = target_rows is not None and (fixed or preplaced)

            if has_target:
                target_x = float(target_rows[i][0])
                target_y = float(target_rows[i][1])
                width = float(target_rows[i][2])
                height = float(target_rows[i][3])
            else:
                target_x = None
                target_y = None
                width = math.sqrt(area)
                height = math.sqrt(area)

            meta.append(
                BlockMeta(
                    index=i,
                    area=area,
                    fixed=fixed,
                    preplaced=preplaced,
                    mib=int(mib_ids[i]),
                    cluster=int(cluster_ids[i]),
                    boundary=int(boundary_codes[i]),
                    width=width,
                    height=height,
                    target_x=target_x,
                    target_y=target_y,
                )
            )

        return meta

    """Solve quadratic placement centers using a Laplacian system."""
    def _solve_laplacian_centers(
        self,
        meta: List[BlockMeta],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> Tuple[List[float], List[float], List[float]]:
        
        # initialization
        cfg = self.config
        n = len(meta)
        dtype = cfg.dtype

        lap = torch.eye(n, dtype=dtype) * cfg.laplacian_reg
        bx = torch.zeros(n, dtype=dtype)
        by = torch.zeros(n, dtype=dtype)
        degree = torch.full((n,), cfg.laplacian_reg, dtype=dtype)

        # weight: block to block
        for edge in self._rows(b2b_connectivity):
            if edge[0] == -1:
                continue

            i = int(edge[0])
            j = int(edge[1])
            weight = max(float(edge[2]), EPS)

            if not (0 <= i < n and 0 <= j < n):
                continue

            lap[i, i] += weight
            lap[j, j] += weight
            lap[i, j] -= weight
            lap[j, i] -= weight
            degree[i] += weight
            degree[j] += weight

        # weight: pin to block
        pin_rows = self._rows(pins_pos)

        for edge in self._rows(p2b_connectivity):
            if edge[0] == -1:
                continue

            pin = int(edge[0])
            block = int(edge[1])
            weight = max(float(edge[2]), EPS)

            if 0 <= block < n and 0 <= pin < len(pin_rows):
                lap[block, block] += weight
                degree[block] += weight
                bx[block] += weight * float(pin_rows[pin][0])
                by[block] += weight * float(pin_rows[pin][1])

        # preplaced blocks anchor
        for block in meta:
            if block.preplaced and block.target_x is not None and block.target_y is not None:
                i = block.index
                anchor = cfg.preplaced_anchor
                lap[i, i] += anchor
                bx[i] += anchor * (block.target_x + block.width / 2.0)
                by[i] += anchor * (block.target_y + block.height / 2.0)

        # solve linear system
        try:
            center_x = torch.linalg.solve(lap, bx)
            center_y = torch.linalg.solve(lap, by)
        except RuntimeError:
            center_x = torch.arange(n, dtype=dtype)
            center_y = torch.zeros(n, dtype=dtype)

        return center_x.tolist(), center_y.tolist(), degree.tolist()

    def _assign_soft_shapes(
        self,
        meta: List[BlockMeta],
        center_x: List[float],
        center_y: List[float],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> None:
        """Assign preferred width and height for movable soft blocks."""

        cfg = self.config
        n = len(meta)

        spread_x = [0.0] * n
        spread_y = [0.0] * n

        for edge in self._rows(b2b_connectivity):
            if edge[0] == -1:
                continue

            i = int(edge[0])
            j = int(edge[1])
            weight = float(edge[2])

            if not (0 <= i < n and 0 <= j < n):
                continue

            dx = abs(center_x[i] - center_x[j]) * weight
            dy = abs(center_y[i] - center_y[j]) * weight

            spread_x[i] += dx
            spread_x[j] += dx
            spread_y[i] += dy
            spread_y[j] += dy

        pin_rows = self._rows(pins_pos)

        for edge in self._rows(p2b_connectivity):
            if edge[0] == -1:
                continue

            pin = int(edge[0])
            block = int(edge[1])
            weight = float(edge[2])

            if 0 <= block < n and 0 <= pin < len(pin_rows):
                spread_x[block] += abs(center_x[block] - float(pin_rows[pin][0])) * weight
                spread_y[block] += abs(center_y[block] - float(pin_rows[pin][1])) * weight

        for block in meta:
            if block.fixed or block.preplaced:
                continue

            ratio = self._preferred_aspect_ratio(
                block=block,
                spread_x=spread_x[block.index],
                spread_y=spread_y[block.index],
            )

            block.width = math.sqrt(block.area * ratio)
            block.height = block.area / block.width

    def _preferred_aspect_ratio(
        self,
        block: BlockMeta,
        spread_x: float,
        spread_y: float,
    ) -> float:
        """Estimate a soft block aspect ratio from constraints and spread."""

        cfg = self.config

        if block.boundary in (4, 8):
            ratio = 0.25
        elif block.boundary in (1, 2):
            ratio = 4.0
        elif block.boundary:
            ratio = 1.0
        else:
            ratio = ((spread_x + 1.0) / (spread_y + 1.0)) ** 0.25

        return min(max(ratio, cfg.min_aspect_ratio), cfg.max_aspect_ratio)

    def _apply_safe_mib_shapes(self, meta: List[BlockMeta]) -> None:
        """Project each MIB group to one common shape when hard-safe."""

        groups = {}
        for block in meta:
            if block.mib:
                groups.setdefault(block.mib, []).append(block)

        for members in groups.values():
            common_shape = self._choose_safe_mib_shape(members)

            if common_shape is None:
                continue

            width, height = common_shape

            for member in members:
                if member.fixed or member.preplaced:
                    continue

                member.width = width
                member.height = height

    def _choose_safe_mib_shape(
        self,
        members: List[BlockMeta],
    ) -> Optional[Tuple[float, float]]:
        """Choose one MIB shape, or return None if no hard-safe shape exists."""

        immutable_members = [
            member for member in members
            if member.fixed or member.preplaced
        ]

        if immutable_members:
            return self._choose_immutable_mib_shape(members, immutable_members)

        return self._choose_soft_only_mib_shape(members)

    def _choose_immutable_mib_shape(
        self,
        members: List[BlockMeta],
        immutable_members: List[BlockMeta],
    ) -> Optional[Tuple[float, float]]:
        """Use immutable shape only if all members can legally share it."""

        width = immutable_members[0].width
        height = immutable_members[0].height

        for member in immutable_members[1:]:
            if abs(member.width - width) > EPS or abs(member.height - height) > EPS:
                return None

        common_area = width * height

        for member in members:
            if member.fixed or member.preplaced:
                continue

            if abs(common_area - member.area) / member.area > AREA_TOL:
                return None

        return width, height

    def _choose_soft_only_mib_shape(
        self,
        members: List[BlockMeta],
    ) -> Optional[Tuple[float, float]]:
        """Choose a common soft-only MIB shape from compatible preferred shapes."""

        sorted_areas = sorted(member.area for member in members)
        ref_area = sorted_areas[len(sorted_areas) // 2]

        for member in members:
            if abs(member.area - ref_area) / ref_area > AREA_TOL:
                return None

        ratios = [
            max(member.width / max(member.height, EPS), EPS)
            for member in members
        ]
        ratios.sort()
        ratio = ratios[len(ratios) // 2]
        ratio = min(max(ratio, self.config.min_aspect_ratio), self.config.max_aspect_ratio)

        width = math.sqrt(ref_area * ratio)
        height = ref_area / width

        common_area = width * height

        for member in members:
            if abs(common_area - member.area) / member.area > AREA_TOL:
                return None

        return width, height

    @staticmethod
    def _centers_to_rectangles(
        meta: List[BlockMeta],
        center_x: List[float],
        center_y: List[float],
    ) -> List[Position]:
        """Convert center coordinates into lower-left rectangle coordinates."""

        positions: List[Position] = []

        for block in meta:
            if block.preplaced and block.target_x is not None and block.target_y is not None:
                x = block.target_x
                y = block.target_y
            else:
                x = center_x[block.index] - block.width / 2.0
                y = center_y[block.index] - block.height / 2.0

            positions.append(
                (
                    float(x),
                    float(y),
                    float(block.width),
                    float(block.height),
                )
            )

        return positions

    @staticmethod
    def _overlap_amount(a: Position, b: Position) -> Tuple[float, float]:
        """Return positive overlap amounts along x and y axes."""

        ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        return max(0.0, ox), max(0.0, oy)

    def _rough_spread(
        self,
        meta: List[BlockMeta],
        positions: List[Position],
    ) -> List[Position]:
        """Apply a light repulsive spreading pass before hard legalization."""

        cfg = self.config
        pos = [list(p) for p in positions]
        n = len(pos)

        for iteration in range(cfg.spread_iters):
            dx = [0.0] * n
            dy = [0.0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    ox, oy = self._overlap_amount(tuple(pos[i]), tuple(pos[j]))

                    if ox <= EPS or oy <= EPS:
                        continue

                    ci_x = pos[i][0] + pos[i][2] / 2.0
                    ci_y = pos[i][1] + pos[i][3] / 2.0
                    cj_x = pos[j][0] + pos[j][2] / 2.0
                    cj_y = pos[j][1] + pos[j][3] / 2.0

                    sign_x = 1.0 if ci_x >= cj_x else -1.0
                    sign_y = 1.0 if ci_y >= cj_y else -1.0

                    if ox <= oy:
                        push = ox * sign_x
                        dx[i] += push
                        dx[j] -= push
                    else:
                        push = oy * sign_y
                        dy[i] += push
                        dy[j] -= push

            weight = cfg.spread_weight_start + cfg.spread_weight_growth * iteration

            for i, block in enumerate(meta):
                if block.preplaced:
                    continue

                pos[i][0] += cfg.spread_step * weight * dx[i]
                pos[i][1] += cfg.spread_step * weight * dy[i]

        return [tuple(p) for p in pos]