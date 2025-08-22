from hex.mcts_agent import MCTSAgent
from hex.mcts_node import MCTSNode
import math


# ===============================
# 3. MCTS avec Progressive Bias
# ===============================
class MCTSProgressiveBiasAgent(MCTSAgent):
    def __init__(self, exploration_constant: float = math.sqrt(2), max_time: float = 1.0, bias_weight: float = 0.1):
        super().__init__(exploration_constant, max_time)
        self.bias_weight = bias_weight

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda child: self._biased_uct(node, child))
        return node

    def _biased_uct(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        q_value = child.wins / child.visits
        exploration = self.exploration_constant * math.sqrt(math.log(parent.visits) / child.visits)

        # Biais heuristique : proche du centre
        center_x, center_y = parent.board.size // 2, parent.board.size // 2
        dist = abs(child.move[0] - center_x) + abs(child.move[1] - center_y)
        bias = self.bias_weight / (1 + dist)

        return q_value + exploration + bias