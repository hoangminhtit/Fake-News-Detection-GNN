#!/usr/bin/env python3
"""
Graph Construction - Create edge_index for PyTorch Geometric
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

# Determine project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


class GraphBuilder:
    """Build graph and create edge_index"""
    
    def __init__(self, data_dir='data', output_dir='output', verbose=True):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.graph_dir = self.data_dir / 'graph'
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.node_id_mapping = None
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def load_data(self):
        """Load preprocessed nodes and edges"""
        nodes_path = self.processed_dir / 'nodes.csv'
        edges_path = self.processed_dir / 'edges.csv'
        
        if not nodes_path.exists() or not edges_path.exists():
            self.log(f"✗ Error: {nodes_path} or {edges_path} not found")
            return False
        
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        self.log(f"✓ Loaded nodes: {self.nodes_df.shape}, edges: {self.edges_df.shape}")
        return True
    
    def build_graph(self):
        """Build NetworkX graph"""
        if self.nodes_df is None or self.edges_df is None:
            self.log("✗ Error: Data not loaded")
            return False
        
        self.G = nx.DiGraph()
        
        # Add edges
        for _, row in self.edges_df.iterrows():
            self.G.add_edge(row['source'], row['target'], weight=row['weight'])
        
        # Add node attributes
        for _, row in self.nodes_df.iterrows():
            if row['node_id'] in self.G.nodes():
                self.G.nodes[row['node_id']]['label'] = row['label']
                if 'title' in row:
                    self.G.nodes[row['node_id']]['title'] = row['title']
        
        # Add isolated nodes
        all_nodes = set(self.nodes_df['node_id'].unique())
        existing_nodes = set(self.G.nodes())
        for node_id in all_nodes - existing_nodes:
            label = self.nodes_df[self.nodes_df['node_id'] == node_id]['label'].values[0]
            self.G.add_node(node_id, label=label)
        
        self.log(f"✓ Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return True
    
    def create_edge_index(self):
        """Create PyTorch edge_index format"""
        if self.G is None:
            self.log("✗ Error: Graph not built")
            return False
        
        unique_nodes = sorted(self.G.nodes())
        self.node_id_mapping = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
        
        edge_list = list(self.G.edges())
        source_indices = np.array([self.node_id_mapping[src] for src, _ in edge_list])
        target_indices = np.array([self.node_id_mapping[tgt] for _, tgt in edge_list])
        
        edge_index = np.array([source_indices, target_indices], dtype=np.int64)
        
        edge_index_path = self.graph_dir / 'edge_index.npy'
        np.save(edge_index_path, edge_index)
        
        self.log(f"✓ Saved edge_index.npy: shape {edge_index.shape}")
        return True
    
    def run(self):
        """Run full process"""
        if not self.load_data():
            return False
        if not self.build_graph():
            return False
        if not self.create_edge_index():
            return False
        return True
