#!/usr/bin/env python3

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer
from legalization_Angela_0626 import Legalization
from analytic_optimizer_0623 import AnalyticOptimizer

Position = Tuple[float, float, float, float]

class MyOptimizer(FloorplanOptimizer):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.analytic_optimizer = AnalyticOptimizer(verbose=verbose)
        self.legalizer = Legalization(verbose=verbose)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Position]:

        if block_count == 0:
            return []

        if target_positions is None:
            target_positions = torch.zeros((block_count, 4), dtype=torch.float64)

        # Step 1: Generate a floorplan as the legalization input.
        analytic_floorplan = self.analytic_optimizer.analytic_solver(
            block_count, 
            area_targets, 
            b2b_connectivity, 
            p2b_connectivity,
            pins_pos, 
            constraints, 
            target_positions
        )
                
        # Step 2: Run the custom legalizer to repair the random floorplan.
        legalized_floorplan = self.legalizer.legalization_checker(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            analytic_floorplan,
            target_positions,
        )
        
        return legalized_floorplan