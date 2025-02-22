from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    nodes.sort(key=lambda x: (sum(cap.memory for cap in x[1]), x[0]), reverse=True)
    total_memory = sum(sum(cap.memory for cap in node[1]) for node in nodes)
    partitions = []
    start = 0
    for node in nodes:
      node_memory = sum(cap.memory for cap in node[1])
      end = round(start + (node_memory/total_memory), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions
