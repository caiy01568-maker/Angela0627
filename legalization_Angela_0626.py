# !/usr/bin/env python3
"""
Usage:
    from legalization_Angela_0626 import Legalization

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.legalizer = Legalization(verbose=verbose)  ## REMEMBER TO ADD THIS LINE IN YOUR OPTIMIZER CLASS


    return self.legalizer.legalization_checker(
        block_count, area_targets, b2b_connectivity, p2b_connectivity,
        pins_pos, constraints, input_floorplan, target_positions
    )
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch

Position = Tuple[float, float, float, float]
Center = Tuple[float, float]
EPS = 1e-9

class Legalization():

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # These weights are intentionally conservative.  Hard legality is still
        # controlled by candidate filtering; weights only rank legal candidates.
        self.w_disp = 1.0 # smaller: legalizer moves blocks further
        self.w_bbox = 8.0 # larger: legalizer prefers smaller bounding box growth
        self.w_b2b = 0.04
        self.w_p2b = 0.05
        self.w_cluster = 3.0
        self.w_boundary = 40.0

        # Limits keep runtime bounded when many edge-aligned candidates exist.
        self.max_x_candidates = 32
        self.max_y_candidates = 32
        self.max_legal_candidates = 24

    def legalization_checker(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        input_floorplan: List[Position],
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Position]:
        
        if block_count == 0:
            return []
                
        # MIB solver, does not break the hard constraints(comment out the following line if you don't need it)
        self._harmonize_mib_shapes(block_count, input_floorplan, constraints, target_positions, area_targets)

        # Preplaced, Fixed blocks and Area targets(comment out the following line if you don't need it)
        self._cover(block_count, input_floorplan, constraints, target_positions, area_targets)

        # Overlaps repulsion(comment out the following lines if you don't need it)
        legalized_floorplan = self._resolve_overlaps(
            block_count,
            input_floorplan,
            constraints,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            target_positions,
            area_targets,
        )

        # Safe soft-boundary cleanup. (comment out the following lines if you don't need it)
        # Accept a snap only when:
        #   1. boundary violation decreases,
        #   2. no overlap is introduced,
        #   3. total HPWL proxy does not increase too much.
        legalized_floorplan = self._boundary_snap(
            legalized_floorplan,
            constraints,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            hpwl_rel_limit=0.01,
        )

        return legalized_floorplan

    # ══════════════════════════════════════════════════════════════════════
    # Preplaced, Fixed blocks and Area targets Legalization
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _cover(block_count, input_floorplan, constraints, target_positions, area_targets) -> None:
        for i in range(block_count):
            x, y, w, h = input_floorplan[i]

            # preplaced
            if float(constraints[i, 1]) != 0.0:        
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            # fixed
            elif float(constraints[i, 0]) != 0.0:      
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            # area
            else :
                current_area = w * h
                target_area = float(area_targets[i])
                if w <= 0 or h <= 0:
                    w = h = math.sqrt(target_area)
                elif abs((current_area - target_area) / target_area) >= 0.01 :
                    ratio = w / h
                    w = math.sqrt(target_area * ratio)
                    h = math.sqrt(target_area / ratio)
                
            input_floorplan[i] = (x, y, w, h) 

    # ══════════════════════════════════════════════════════════════════════
    # MIB Solver
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _harmonize_mib_shapes(
        block_count: int,
        input_floorplan: List[Position],
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        area_targets: torch.Tensor,
    ) -> None:
        """Group MIB blocks by same area and similar aspect ratio."""
        area_rel_tol = 1e-6
        aspect_tol_log = math.log(1.15)

        mib_groups = {}
        for i in range(block_count):
            gid = int(float(constraints[i, 2]))
            if gid > 0:
                mib_groups.setdefault(gid, []).append(i)

        for members in mib_groups.values():
            if len(members) < 2:
                continue

            # Step 1: split by target area.
            area_groups = []
            for i in members:
                area_i = max(float(area_targets[i]), EPS)
                inserted = False

                for group in area_groups:
                    ref_area = max(float(area_targets[group[0]]), EPS)
                    rel_error = abs(area_i - ref_area) / max(area_i, ref_area, EPS)

                    if rel_error <= area_rel_tol:
                        group.append(i)
                        inserted = True
                        break

                if not inserted:
                    area_groups.append([i])

            # Step 2: split each same-area group by current aspect ratio.
            for area_group in area_groups:
                if len(area_group) < 2:
                    continue

                block_logs = []
                for i in area_group:
                    fixed = float(constraints[i, 0]) != 0.0
                    preplaced = float(constraints[i, 1]) != 0.0

                    if (fixed or preplaced) and target_positions is not None:
                        w = float(target_positions[i, 2])
                        h = float(target_positions[i, 3])
                    else:
                        _, _, w, h = input_floorplan[i]

                    if w <= 0.0 or h <= 0.0:
                        log_ratio = 0.0
                    else:
                        log_ratio = math.log(max(float(w), EPS) / max(float(h), EPS))

                    block_logs.append((log_ratio, i))

                block_logs.sort(key=lambda item: item[0])

                subgroups = []
                subgroup_refs = []

                for log_ratio, i in block_logs:
                    inserted = False

                    for k, ref_log in enumerate(subgroup_refs):
                        if abs(log_ratio - ref_log) <= aspect_tol_log:
                            subgroups[k].append(i)

                            # Update subgroup reference by running average.
                            n = len(subgroups[k])
                            subgroup_refs[k] = ((n - 1) * ref_log + log_ratio) / n

                            inserted = True
                            break

                    if not inserted:
                        subgroups.append([i])
                        subgroup_refs.append(log_ratio)

                # Step 3: harmonize each subgroup.
                for subgroup, ref_log in zip(subgroups, subgroup_refs):
                    if len(subgroup) < 2:
                        continue

                    immutable_shapes = []
                    for i in subgroup:
                        fixed = float(constraints[i, 0]) != 0.0
                        preplaced = float(constraints[i, 1]) != 0.0

                        if (fixed or preplaced) and target_positions is not None:
                            tw = float(target_positions[i, 2])
                            th = float(target_positions[i, 3])

                            if tw > 0.0 and th > 0.0:
                                immutable_shapes.append((tw, th))

                    if immutable_shapes:
                        ref_w, ref_h = immutable_shapes[0]

                        has_conflict = any(
                            abs(w - ref_w) > 1e-5 or abs(h - ref_h) > 1e-5
                            for w, h in immutable_shapes[1:]
                        )
                        if has_conflict:
                            continue

                        ref_ratio = max(ref_w, EPS) / max(ref_h, EPS)
                    else:
                        ref_ratio = math.exp(ref_log)

                    # Apply common ratio only to movable soft blocks.
                    # _cover() later enforces exact area and immutable geometry.
                    for i in subgroup:
                        fixed = float(constraints[i, 0]) != 0.0
                        preplaced = float(constraints[i, 1]) != 0.0

                        if fixed or preplaced:
                            continue

                        x, y, _, _ = input_floorplan[i]
                        area = max(float(area_targets[i]), EPS)
                        new_w = math.sqrt(area * ref_ratio)
                        new_h = math.sqrt(area / ref_ratio)
                        input_floorplan[i] = (x, y, new_w, new_h)

    # ══════════════════════════════════════════════════════════════════════
    # Overlap Legalization
    # ══════════════════════════════════════════════════════════════════════  
    @staticmethod
    def _resolve_overlaps(
        block_count,
        input_floorplan,
        constraints,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        target_positions,
        area_targets,
    ) -> List[Position]:

        preplaced = { i for i in range(block_count) if float(constraints[i, 1]) != 0.0}
        movable = [i for i in range(block_count) if i not in preplaced]

        # Place blocks closer to the shifted origin first.
        min_x = min(r[0] for r in input_floorplan)
        min_y = min(r[1] for r in input_floorplan)
        order = sorted(
            movable,
            key=lambda i: Legalization._placement_order_key(
                i, 
                input_floorplan, 
                min_x, 
                min_y, 
                constraints, 
                b2b_connectivity, 
                p2b_connectivity,
                area_targets
            )
        )

        return Legalization._place(
            input_floorplan,
            preplaced,
            order,
            constraints,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
        )

    @staticmethod
    def _placement_order_key(
        i: int,
        input_floorplan: List[Position],
        min_x: float,
        min_y: float,
        constraints: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        area_targets,
    ) -> tuple:
        """Place important and difficult blocks earlier."""

        x, y, w, h = input_floorplan[i]
        area = max(w * h, 1e-9)

        shifted_x = x - min_x
        shifted_y = y - min_y
        dist2 = shifted_x * shifted_x + shifted_y * shifted_y

        total_area = max(float(torch.sum(area_targets)), 1e-9)
        avg_area = total_area / max(len(input_floorplan), 1)

        # Use average block area as distance-square bucket size.
        # Blocks in nearby distance ranges will then compare degree and area.
        dist_bucket_size = max(avg_area, 1e-9)
        dist_bucket = math.floor(dist2 / dist_bucket_size)

        # constraints is always [fixed, preplaced, MIB_id, grouping_id, boundary_mask]
        boundary_mask = int(float(constraints[i, 4]))

        left_bit = 1 if (boundary_mask & 1) else 0
        right_bit = 1 if (boundary_mask & 2) else 0
        top_bit = 1 if (boundary_mask & 4) else 0
        bottom_bit = 1 if (boundary_mask & 8) else 0

        grouping_flag = 1.0 if int(float(constraints[i, 3])) != 0 else 0.0

        degree = 0.0
        total_degree = 0.0

        # b2b_connectivity rows are [block_a, block_b, weight].
        for e in range(b2b_connectivity.shape[0]):
            block_a = int(float(b2b_connectivity[e, 0]))
            block_b = int(float(b2b_connectivity[e, 1]))
            weight = abs(float(b2b_connectivity[e, 2]))

            total_degree += weight
            if block_a == i or block_b == i:
                degree += weight

        # p2b_connectivity rows are [pin_id, block_id, weight].
        for e in range(p2b_connectivity.shape[0]):
            block_id = int(float(p2b_connectivity[e, 1]))
            weight = abs(float(p2b_connectivity[e, 2]))

            total_degree += weight
            if block_id == i:
                degree += weight

        total_degree = max(total_degree, 1e-9)

        heavy = degree / total_degree + area / total_area

        return (
            (right_bit + top_bit),  # right/top boundary blocks last
            -(left_bit + bottom_bit),  # left/bottom boundary blocks first
            -grouping_flag,     # grouping blocks first
            dist_bucket,        # blocks with similar dist2 are tied
            -heavy,             # then compare degree and area
            shifted_y,
            shifted_x,
        )

    @staticmethod
    def _place(
        input_floorplan: List[Position],
        preplaced: set,
        order: List[int],
        constraints: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[Position]:

        result_floorplan = list(input_floorplan)
        placed_ids = sorted(preplaced)
        placed = [result_floorplan[i] for i in placed_ids]

        left_edges = {x for x, y, w, h in placed}
        right_edges = {x + w for x, y, w, h in placed}
        bottom_edges = {y for x, y, w, h in placed}
        top_edges = {y + h for x, y, w, h in placed}

        cell_size = Legalization._grid_cell_size(input_floorplan)
        grid = {}
        for j, rect in enumerate(placed):
            Legalization._grid_add(grid, rect, j, cell_size)

        eps = 1e-9
        for i in order:  # order does not include preplaced
            x0, y0, w0, h0 = result_floorplan[i]
            boundary = int(float(constraints[i, 4]))
            group_id = int(float(constraints[i, 3]))

            old_r = max(right_edges, default=0.0)
            old_t = max(top_edges, default=0.0)

            xs = ({0.0, x0} | left_edges | right_edges | {x - w0 for x in left_edges} | {x - w0 for x in right_edges})
            ys = ({0.0, y0} | bottom_edges | top_edges | {y - h0 for y in bottom_edges} | {y - h0 for y in top_edges})
            
            same_group_xs = set()
            same_group_ys = set()
            same_group_rects = []

            if group_id != 0:
                for rect, other_id in zip(placed, placed_ids):
                    if int(float(constraints[other_id, 3])) != group_id:
                        continue

                    px, py, pw, ph = rect
                    same_group_rects.append(rect)

                    same_group_xs |= {px - w0, px + pw, px, px + pw - w0}
                    same_group_ys |= {py - h0, py + ph, py, py + ph - h0}

            x_candidates = sorted(
                (x for x in xs if x >= 0.0),
                key=lambda x: (
                    0 if ((boundary & 1) and abs(x) <= eps) else 1,
                    0 if ((boundary & 2) and abs((x + w0) - old_r) <= eps) else 1,
                    0 if (x in same_group_xs) else 1,
                    (x - x0) ** 2,
                )
            )

            y_candidates = sorted(
                (y for y in ys if y >= 0.0),
                key=lambda y: (
                    0 if ((boundary & 8) and abs(y) <= eps) else 1,
                    0 if ((boundary & 4) and abs((y + h0) - old_t) <= eps) else 1,
                    0 if (y in same_group_ys) else 1,
                    (y - y0) ** 2,
                )
            )

            legal_candidates = []
                
            c = (x_candidates[0], y_candidates[0], w0, h0)
            if not Legalization._overlaps_grid(c, placed, grid, cell_size):
                legal_candidates.append(c)

            idx = 0
            idy = 0

            while idx + 1 < len(x_candidates) or idy + 1 < len(y_candidates):

                if idx + 1 < len(x_candidates):
                    idx += 1
                    for iy in range(idy + 1):
                        c = (x_candidates[idx], y_candidates[iy], w0, h0)
                        if not Legalization._overlaps_grid(c, placed, grid, cell_size):
                            legal_candidates.append(c)

                if idy + 1 < len(y_candidates):
                    idy += 1
                    for ix in range(idx + 1):
                        c = (x_candidates[ix], y_candidates[idy], w0, h0)
                        if not Legalization._overlaps_grid(c, placed, grid, cell_size):
                            legal_candidates.append(c)

                if len(legal_candidates) >= 4:
                    break

            if not legal_candidates:
                raise RuntimeError(f"No legal candidate found for block {i}")
                
            legal_candidates = legal_candidates[:8]
            
            best = min(
                legal_candidates,
                key=lambda c: Legalization._candidate_score(
                    c,
                    x0,
                    y0,
                    old_r,
                    old_t,
                    boundary,
                    same_group_rects,
                )
            )

            result_floorplan[i] = best
            placed.append(best)
            placed_ids.append(i)
            Legalization._grid_add(grid, best, len(placed) - 1, cell_size)

            bx, by, bw, bh = best
            left_edges.add(bx)
            right_edges.add(bx + bw)
            bottom_edges.add(by)
            top_edges.add(by + bh)

        return result_floorplan

    @staticmethod
    def _candidate_score(
        c: Position,
        x0: float,
        y0: float,
        old_r: float,
        old_t: float,
        boundary: int,
        same_group_rects: List[Position],
    ) -> tuple:
        """Rank by boundary legality, cluster abutment, then bbox growth and displacement."""

        x, y, w, h = c

        eps = 1e-6

        boundary_violation = (
            int((boundary & 1) != 0 and abs(x) > eps) +
            int((boundary & 8) != 0 and abs(y) > eps) +
            int((boundary & 2) != 0 and (x + w) < old_r - eps) +
            int((boundary & 4) != 0 and (y + h) < old_t - eps)
        )

        cluster_violation = 0

        if same_group_rects:
            cluster_violation = 1

            for px, py, pw, ph in same_group_rects:
                if (
                    min(y + h, py + ph) > max(y, py) + eps and
                    (abs((x + w) - px) <= eps or abs((px + pw) - x) <= eps)
                ) or (
                    min(x + w, px + pw) > max(x, px) + eps and
                    (abs((y + h) - py) <= eps or abs((py + ph) - y) <= eps)
                ):
                    cluster_violation = 0
                    break

        scale = math.sqrt(max(w * h, 1e-9))

        grow = (
            max(old_r, x + w) - old_r +
            max(old_t, y + h) - old_t
        ) / scale

        disp = ((x - x0) ** 2 + (y - y0) ** 2) / (scale * scale)

        return (
            boundary_violation,
            cluster_violation,
            grow + 0.1 * disp,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Boundary Snap Legalization
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _boundary_snap(
        floorplan: List[Position],
        constraints: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        hpwl_rel_limit: float = 0.01,
    ) -> List[Position]:
        """Safely snap boundary blocks if boundary miss decreases and HPWL is protected."""

        result = list(floorplan)
        n = len(result)
        eps = 1e-6

        if n == 0:
            return result

        final_l = min(x for x, y, w, h in result)
        final_b = min(y for x, y, w, h in result)
        final_r = max(x + w for x, y, w, h in result)
        final_t = max(y + h for x, y, w, h in result)

        hpwl = 0.0

        for e in range(b2b_connectivity.shape[0]):
            a = int(float(b2b_connectivity[e, 0]))
            b = int(float(b2b_connectivity[e, 1]))
            weight = abs(float(b2b_connectivity[e, 2]))

            if 0 <= a < n and 0 <= b < n:
                ax, ay, aw, ah = result[a]
                bx, by, bw, bh = result[b]
                hpwl += weight * (
                    abs((ax + 0.5 * aw) - (bx + 0.5 * bw)) +
                    abs((ay + 0.5 * ah) - (by + 0.5 * bh))
                )

        if pins_pos is not None:
            pin_count = int(pins_pos.shape[0]) if len(pins_pos.shape) >= 2 else 0

            for e in range(p2b_connectivity.shape[0]):
                pin_id = int(float(p2b_connectivity[e, 0]))
                block_id = int(float(p2b_connectivity[e, 1]))
                weight = abs(float(p2b_connectivity[e, 2]))

                if 0 <= block_id < n and 0 <= pin_id < pin_count:
                    x, y, w, h = result[block_id]
                    hpwl += weight * (
                        abs((x + 0.5 * w) - float(pins_pos[pin_id, 0])) +
                        abs((y + 0.5 * h) - float(pins_pos[pin_id, 1]))
                    )

        hpwl_limit = hpwl * (1.0 + hpwl_rel_limit) + 1e-9

        for i in sorted(
            range(n),
            key=lambda k: (
                -int((int(float(constraints[k, 4])) & 2) != 0)
                -int((int(float(constraints[k, 4])) & 4) != 0),
                -int((int(float(constraints[k, 4])) & 1) != 0)
                -int((int(float(constraints[k, 4])) & 8) != 0),
                k,
            )
        ):
            boundary = int(float(constraints[i, 4]))

            if boundary == 0 or float(constraints[i, 1]) != 0.0:
                continue

            x, y, w, h = result[i]

            old_miss = (
                int((boundary & 1) != 0 and abs(x - final_l) > eps) +
                int((boundary & 2) != 0 and abs((x + w) - final_r) > eps) +
                int((boundary & 4) != 0 and abs((y + h) - final_t) > eps) +
                int((boundary & 8) != 0 and abs(y - final_b) > eps)
            )

            if old_miss == 0:
                continue

            nx = final_l if (boundary & 1) else final_r - w if (boundary & 2) else x
            ny = final_b if (boundary & 8) else final_t - h if (boundary & 4) else y
            nx = 0.0 if abs(nx) <= eps else nx
            ny = 0.0 if abs(ny) <= eps else ny

            new_miss = (
                int((boundary & 1) != 0 and abs(nx - final_l) > eps) +
                int((boundary & 2) != 0 and abs((nx + w) - final_r) > eps) +
                int((boundary & 4) != 0 and abs((ny + h) - final_t) > eps) +
                int((boundary & 8) != 0 and abs(ny - final_b) > eps)
            )

            if new_miss >= old_miss:
                continue

            has_overlap = False

            for j, (px, py, pw, ph) in enumerate(result):
                if j == i:
                    continue

                if (
                    nx < px + pw - eps and
                    nx + w > px + eps and
                    ny < py + ph - eps and
                    ny + h > py + eps
                ):
                    has_overlap = True
                    break

            if has_overlap:
                continue

            old_cx = x + 0.5 * w
            old_cy = y + 0.5 * h
            new_cx = nx + 0.5 * w
            new_cy = ny + 0.5 * h
            old_wire = 0.0
            new_wire = 0.0

            for e in range(b2b_connectivity.shape[0]):
                a = int(float(b2b_connectivity[e, 0]))
                b = int(float(b2b_connectivity[e, 1]))
                weight = abs(float(b2b_connectivity[e, 2]))

                if a == i and 0 <= b < n:
                    ox, oy, ow, oh = result[b]
                elif b == i and 0 <= a < n:
                    ox, oy, ow, oh = result[a]
                else:
                    continue

                ocx = ox + 0.5 * ow
                ocy = oy + 0.5 * oh
                old_wire += weight * (abs(old_cx - ocx) + abs(old_cy - ocy))
                new_wire += weight * (abs(new_cx - ocx) + abs(new_cy - ocy))

            if pins_pos is not None:
                pin_count = int(pins_pos.shape[0]) if len(pins_pos.shape) >= 2 else 0

                for e in range(p2b_connectivity.shape[0]):
                    pin_id = int(float(p2b_connectivity[e, 0]))
                    block_id = int(float(p2b_connectivity[e, 1]))
                    weight = abs(float(p2b_connectivity[e, 2]))

                    if block_id == i and 0 <= pin_id < pin_count:
                        px = float(pins_pos[pin_id, 0])
                        py = float(pins_pos[pin_id, 1])
                        old_wire += weight * (abs(old_cx - px) + abs(old_cy - py))
                        new_wire += weight * (abs(new_cx - px) + abs(new_cy - py))

            if hpwl - old_wire + new_wire <= hpwl_limit:
                result[i] = (nx, ny, w, h)
                hpwl = hpwl - old_wire + new_wire

        return result

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _overlaps(rect: Position, placed: List[Position]) -> bool:
        x, y, w, h = rect
        return any(
            min(x + w, px + pw) > max(x, px) and
            min(y + h, py + ph) > max(y, py)
            for px, py, pw, ph in placed
        )

    @staticmethod
    def _grid_cell_size(input_floorplan: List[Position]) -> float:
        sizes = sorted(
            math.sqrt(max(w * h, 1e-9))
            for x, y, w, h in input_floorplan
            if w > 0.0 and h > 0.0
        )
        return max(sizes[len(sizes) // 2] if sizes else 1.0, 1e-9)

    @staticmethod
    def _grid_cells(rect: Position, cell_size: float):
        x, y, w, h = rect

        eps = 1e-9
        gx0 = math.floor(x / cell_size)
        gx1 = math.floor((x + max(w, eps) - eps) / cell_size)
        gy0 = math.floor(y / cell_size)
        gy1 = math.floor((y + max(h, eps) - eps) / cell_size)

        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                yield (gx, gy)

    @staticmethod
    def _grid_add(grid, rect: Position, rect_id: int, cell_size: float) -> None:
        for cell in Legalization._grid_cells(rect, cell_size):
            grid.setdefault(cell, []).append(rect_id)

    @staticmethod
    def _overlaps_grid(
        rect: Position,
        placed: List[Position],
        grid,
        cell_size: float,
    ) -> bool:
        x, y, w, h = rect
        seen = set()

        for cell in Legalization._grid_cells(rect, cell_size):
            for j in grid.get(cell, []):
                if j in seen:
                    continue

                seen.add(j)
                px, py, pw, ph = placed[j]

                if (
                    x < px + pw and
                    x + w > px and
                    y < py + ph and
                    y + h > py
                ):
                    return True

        return False
