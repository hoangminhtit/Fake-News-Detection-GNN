#!/usr/bin/env python3
"""
Graph Visualization - Create visualization plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from pathlib import Path

# Determine project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class GraphVisualizer:
    """Create graph visualizations"""
    
    def __init__(self, graph, nodes_df, viz_dir='output/visualizations', verbose=True):
        self.G = graph
        self.nodes_df = nodes_df
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def visualize_all(self):
        """Create all visualizations"""
        if self.G is None:
            self.log("✗ Error: Graph not available")
            return False
        
        self.log("Creating visualizations...")
        try:
            self._degree_distribution()
            self._in_out_degree()
            self._label_distribution()
            self._graph_sample()
            self.log("✓ Visualizations completed\n")
            return True
        except Exception as e:
            self.log(f"✗ Error: {e}")
            return False
    
    def _degree_distribution(self):
        """Degree distribution plot"""
        degrees = np.array([self.G.degree(n) for n in self.G.nodes()])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'degree_distribution.png', dpi=150)
        plt.close()
        self.log("  ✓ degree_distribution.png")
    
    def _in_out_degree(self):
        """In-degree vs Out-degree plot"""
        in_deg = np.array([self.G.in_degree(n) for n in self.G.nodes()])
        out_deg = np.array([self.G.out_degree(n) for n in self.G.nodes()])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(in_deg, out_deg, alpha=0.5, s=20, color='coral')
        ax.set_xlabel('In-Degree')
        ax.set_ylabel('Out-Degree')
        ax.set_title('In-Degree vs Out-Degree')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'indegree_outdegree.png', dpi=150)
        plt.close()
        self.log("  ✓ indegree_outdegree.png")
    
    def _label_distribution(self):
        """Label distribution plot"""
        fake = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 1)
        real = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Real', 'Fake']
        counts = [real, fake]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/self.G.number_of_nodes()*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'label_distribution.png', dpi=150)
        plt.close()
        self.log("  ✓ label_distribution.png")
    
    def _graph_sample(self):
        """Graph network visualization"""
        sample_size = min(200, self.G.number_of_nodes())
        if sample_size < self.G.number_of_nodes():
            nodes = sorted(self.G.nodes(), key=lambda x: self.G.degree(x), reverse=True)[:sample_size]
            G_sample = self.G.subgraph(nodes)
        else:
            G_sample = self.G
        
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G_sample, k=0.5, iterations=50, seed=42)
        
        colors = ['#e74c3c' if G_sample.nodes[n].get('label') == 1 else '#2ecc71' 
                  for n in G_sample.nodes()]
        
        nx.draw_networkx_nodes(G_sample, pos, node_color=colors, node_size=100, ax=ax, alpha=0.7)
        nx.draw_networkx_edges(G_sample, pos, ax=ax, edge_color='gray', arrows=True, 
                               arrowsize=10, alpha=0.3, connectionstyle='arc3,rad=0.1', width=0.5)
        
        ax.set_title(f'Graph Visualization ({sample_size} nodes)')
        ax.axis('off')
        
        from matplotlib.patches import Patch
        leg = [Patch(facecolor='#2ecc71', alpha=0.7, label='Real'),
               Patch(facecolor='#e74c3c', alpha=0.7, label='Fake')]
        ax.legend(handles=leg, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'graph_visualization.png', dpi=150)
        plt.close()
        self.log("  ✓ graph_visualization.png")
