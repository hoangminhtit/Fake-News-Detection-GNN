#!/usr/bin/env python3
"""
Graph Construction Module for Fake News Detection Project
Member 2 - Graph Mining Implementation
"""

import argparse
import pandas as pd
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class GraphBuilder:
    """Main class for building and analyzing propagation graphs"""
    
    def __init__(self, data_dir='data', output_dir='output', verbose=True):
        """Initialize graph builder"""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'  # nodes.csv, edges.csv
        self.graph_dir = self.data_dir / 'graph'  # edge_index.npy output
        self.viz_dir = Path(output_dir) / 'visualizations'  # PNG files
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        
        # Data structures
        self.nodes_df = None
        self.edges_df = None
        self.G = None  # NetworkX DiGraph
        self.node_id_mapping = None  # Original node_id → index mapping
        self.reverse_mapping = None  # index → original node_id
        
        self.log_messages = []
        
    def log(self, msg):
        """Print and store log messages"""
        if self.verbose:
            print(msg)
        self.log_messages.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    
    # ==================== Task 1: Load Data ====================
    def load_data(self):
        """Load nodes.csv and edges.csv from processed directory"""
        self.log("=" * 60)
        self.log("TASK 1: Loading Data")
        self.log("=" * 60)
        
        try:
            nodes_path = self.processed_dir / 'nodes.csv'
            edges_path = self.processed_dir / 'edges.csv'
            
            if not nodes_path.exists():
                raise FileNotFoundError(f"File not found: {nodes_path}")
            if not edges_path.exists():
                raise FileNotFoundError(f"File not found: {edges_path}")
            
            self.nodes_df = pd.read_csv(nodes_path)
            self.edges_df = pd.read_csv(edges_path)
            
            self.log(f"✓ Loaded nodes.csv: shape {self.nodes_df.shape}")
            self.log(f"✓ Loaded edges.csv: shape {self.edges_df.shape}")
            
            # Data overview
            self.log(f"\nNodes columns: {list(self.nodes_df.columns)}")
            self.log(f"Edges columns: {list(self.edges_df.columns)}")
            
            # Statistics
            self.log(f"\nNodes statistics:")
            self.log(f"  - Total nodes: {len(self.nodes_df)}")
            self.log(f"  - Fake (label=1): {(self.nodes_df['label'] == 1).sum()}")
            self.log(f"  - Real (label=0): {(self.nodes_df['label'] == 0).sum()}")
            self.log(f"  - Fake percentage: {(self.nodes_df['label'] == 1).sum() / len(self.nodes_df) * 100:.1f}%")
            
            self.log(f"\nEdges statistics:")
            self.log(f"  - Total edges: {len(self.edges_df)}")
            
            # Check for issues
            missing_nodes = set(self.edges_df['source'].unique()) | set(self.edges_df['target'].unique())
            missing_nodes = missing_nodes - set(self.nodes_df['node_id'].unique())
            if missing_nodes:
                self.log(f"  ⚠ WARNING: {len(missing_nodes)} nodes in edges not found in nodes.csv")
                self.log(f"    Missing node IDs: {sorted(list(missing_nodes))[:10]}...")
            
            # Check for duplicates
            duplicate_edges = self.edges_df.duplicated(subset=['source', 'target']).sum()
            if duplicate_edges > 0:
                self.log(f"  ⚠ WARNING: {duplicate_edges} duplicate edges found")
            
            # Check for null values
            null_nodes = self.nodes_df.isnull().sum()
            null_edges = self.edges_df.isnull().sum()
            if null_nodes.sum() > 0:
                self.log(f"  ⚠ WARNING: Null values in nodes_df: {null_nodes.sum()}")
            if null_edges.sum() > 0:
                self.log(f"  ⚠ WARNING: Null values in edges_df: {null_edges.sum()}")
            
            self.log("\n✓ Data loading completed successfully\n")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {e}")
            return False
    
    # ==================== Task 2: Build Graph ====================
    def build_graph(self):
        """Build NetworkX DiGraph from edges and add node/edge attributes"""
        self.log("=" * 60)
        self.log("TASK 2: Building Propagation Graph")
        self.log("=" * 60)
        
        if self.nodes_df is None or self.edges_df is None:
            self.log("✗ ERROR: Data not loaded. Run load_data() first.")
            return False
        
        try:
            # Create directed graph
            self.G = nx.DiGraph()
            
            # Add edges with weight attribute
            for _, row in self.edges_df.iterrows():
                self.G.add_edge(
                    row['source'], 
                    row['target'], 
                    weight=row['weight']
                )
            
            # Add node attributes (label, title)
            for _, row in self.nodes_df.iterrows():
                if row['node_id'] in self.G.nodes():
                    self.G.nodes[row['node_id']]['label'] = row['label']
                    self.G.nodes[row['node_id']]['title'] = row['title']
            
            # Add isolated nodes (nodes without edges)
            all_nodes = set(self.nodes_df['node_id'].unique())
            existing_nodes = set(self.G.nodes())
            isolated = all_nodes - existing_nodes
            
            for node_id in isolated:
                label = self.nodes_df[self.nodes_df['node_id'] == node_id]['label'].values[0]
                title = self.nodes_df[self.nodes_df['node_id'] == node_id]['title'].values[0]
                self.G.add_node(node_id, label=label, title=title)
            
            # Statistics
            self.log(f"✓ Graph created:")
            self.log(f"  - Nodes: {self.G.number_of_nodes()}")
            self.log(f"  - Edges: {self.G.number_of_edges()}")
            self.log(f"  - Density: {nx.density(self.G):.4f}")
            
            # Connected components
            weakly_connected = nx.number_weakly_connected_components(self.G)
            strongly_connected = nx.number_strongly_connected_components(self.G)
            self.log(f"  - Weakly connected components: {weakly_connected}")
            self.log(f"  - Strongly connected components: {strongly_connected}")
            
            # Degree statistics
            degrees = [self.G.degree(n) for n in self.G.nodes()]
            in_degrees = [self.G.in_degree(n) for n in self.G.nodes()]
            out_degrees = [self.G.out_degree(n) for n in self.G.nodes()]
            
            self.log(f"\nDegree statistics:")
            self.log(f"  - Mean degree: {np.mean(degrees):.2f}")
            self.log(f"  - Max degree: {np.max(degrees)}")
            self.log(f"  - Min degree: {np.min(degrees)}")
            
            self.log(f"\nIn-degree statistics:")
            self.log(f"  - Mean: {np.mean(in_degrees):.2f}")
            self.log(f"  - Max: {np.max(in_degrees)}")
            
            self.log(f"\nOut-degree statistics:")
            self.log(f"  - Mean: {np.mean(out_degrees):.2f}")
            self.log(f"  - Max: {np.max(out_degrees)}")
            
            # Label distribution
            fake_count = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 1)
            real_count = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 0)
            self.log(f"\nNode labels in graph:")
            self.log(f"  - Fake nodes: {fake_count}")
            self.log(f"  - Real nodes: {real_count}")
            
            self.log("\n✓ Graph building completed successfully\n")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== Task 3: Create Edge Index ====================
    def create_edge_index(self):
        """Create edge_index tensor for PyTorch/PyTorch Geometric"""
        self.log("=" * 60)
        self.log("TASK 3: Creating Edge Index (PyTorch Format)")
        self.log("=" * 60)
        
        if self.G is None:
            self.log("✗ ERROR: Graph not built. Run build_graph() first.")
            return False
        
        try:
            # Get all unique nodes from graph (sorted)
            unique_nodes = sorted(self.G.nodes())
            
            # Create mapping: original_node_id -> index
            self.node_id_mapping = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
            self.reverse_mapping = {idx: node_id for node_id, idx in self.node_id_mapping.items()}
            
            self.log(f"Created mapping for {len(self.node_id_mapping)} nodes")
            
            # Get edges
            edge_list = list(self.G.edges())
            
            # Map edges to indices
            source_indices = np.array([self.node_id_mapping[src] for src, _ in edge_list])
            target_indices = np.array([self.node_id_mapping[tgt] for _, tgt in edge_list])
            
            # Create edge_index tensor [2, num_edges]
            edge_index = np.array([source_indices, target_indices], dtype=np.int64)
            
            # Save edge_index.npy to graph directory
            edge_index_path = self.graph_dir / 'edge_index.npy'
            np.save(edge_index_path, edge_index)
            self.log(f"✓ Saved edge_index.npy: shape {edge_index.shape}")
            
            self.log(f"\nEdge index statistics:")
            self.log(f"  - Num unique nodes: {len(unique_nodes)}")
            self.log(f"  - Num edges: {edge_index.shape[1]}")
            self.log(f"  - Edge index shape: {edge_index.shape}")
            
            self.log("\n✓ Edge index creation completed successfully\n")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== Task 4: Visualize Graph ====================
    def visualize(self):
        """Create visualizations for graph analysis"""
        self.log("=" * 60)
        self.log("TASK 4: Creating Visualizations")
        self.log("=" * 60)
        
        if self.G is None:
            self.log("✗ ERROR: Graph not built.")
            return False
        
        try:
            # Get degree information
            degrees = np.array([self.G.degree(n) for n in self.G.nodes()])
            in_degrees = np.array([self.G.in_degree(n) for n in self.G.nodes()])
            out_degrees = np.array([self.G.out_degree(n) for n in self.G.nodes()])
            
            # ---- Visualization 1: Degree Distribution ----
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_xlabel('Degree', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'degree_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Saved degree_distribution.png")
            
            # ---- Visualization 2: In-Degree vs Out-Degree ----
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.scatter(in_degrees, out_degrees, alpha=0.5, s=20, color='coral')
            ax.set_xlabel('In-Degree', fontsize=12)
            ax.set_ylabel('Out-Degree', fontsize=12)
            ax.set_title('In-Degree vs Out-Degree', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'indegree_outdegree.png', dpi=150, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Saved indegree_outdegree.png")
            
            # ---- Visualization 3: Label Distribution ----
            fake_count = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 1)
            real_count = sum(1 for _, attr in self.G.nodes(data=True) if attr.get('label') == 0)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            labels = ['Real', 'Fake']
            counts = [real_count, fake_count]
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Node Label Distribution', fontsize=14, fontweight='bold')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}\n({count/self.G.number_of_nodes()*100:.1f}%)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'label_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Saved label_distribution.png")
            
            # ---- Visualization 4: Graph Sample (spring layout) ----
            # For large graphs, sample a subset of nodes
            sample_size = min(200, self.G.number_of_nodes())
            if sample_size < self.G.number_of_nodes():
                # Sample nodes by degree (include hub nodes)
                nodes_by_degree = sorted(self.G.nodes(), key=lambda x: self.G.degree(x), reverse=True)
                sampled_nodes = set(nodes_by_degree[:sample_size])
                G_sample = self.G.subgraph(sampled_nodes)
                self.log(f"✓ Sampling {sample_size} nodes from {self.G.number_of_nodes()} nodes for visualization")
            else:
                G_sample = self.G
            
            try:
                fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                
                # Layout
                pos = nx.spring_layout(G_sample, k=0.5, iterations=50, seed=42)
                
                # Node colors by label
                node_colors = []
                for node in G_sample.nodes():
                    label = G_sample.nodes[node].get('label', 0)
                    node_colors.append('#e74c3c' if label == 1 else '#2ecc71')  # Red for fake, green for real
                
                # Draw network
                nx.draw_networkx_nodes(G_sample, pos, node_color=node_colors, 
                                      node_size=100, ax=ax, alpha=0.7)
                nx.draw_networkx_edges(G_sample, pos, ax=ax, edge_color='gray', 
                                      arrows=True, arrowsize=10, alpha=0.3, 
                                      connectionstyle='arc3,rad=0.1', width=0.5)
                
                ax.set_title(f'Propagation Graph Visualization (Sample: {sample_size} nodes)', 
                           fontsize=14, fontweight='bold')
                ax.axis('off')
                
                # Legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', alpha=0.7, label='Real News'),
                    Patch(facecolor='#e74c3c', alpha=0.7, label='Fake News')
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
                
                plt.tight_layout()
                plt.savefig(self.viz_dir / 'graph_visualization.png', dpi=150, bbox_inches='tight')
                plt.close()
                self.log(f"✓ Saved graph_visualization.png")
            except Exception as e:
                self.log(f"  ⚠ WARNING: Could not create graph visualization: {e}")
            
            self.log("\n✓ Visualization completed successfully\n")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== Task 5: Advanced Analysis ====================
    def analyze_advanced(self):
        """Advanced graph analysis (PageRank, centrality measures)"""
        self.log("=" * 60)
        self.log("TASK 5: Advanced Graph Analysis")
        self.log("=" * 60)
        
        if self.G is None or self.node_id_mapping is None:
            self.log("✗ ERROR: Graph or mapping not available.")
            return False
        
        try:
            num_nodes = len(self.node_id_mapping)
            
            # ---- PageRank ----
            self.log("Computing PageRank...")
            pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=100)
            pagerank_array = np.array([pagerank[self.reverse_mapping[i]] for i in range(num_nodes)])
            np.save(self.data_dir / 'pagerank.npy', pagerank_array)
            self.log(f"✓ Saved pagerank.npy")
            
            # ---- Centrality Measures ----
            self.log("Computing centrality measures...")
            
            # In-degree centrality
            in_centrality = nx.in_degree_centrality(self.G)
            # Out-degree centrality
            out_centrality = nx.out_degree_centrality(self.G)
            # Betweenness centrality (subset for efficiency)
            try:
                betweenness = nx.betweenness_centrality(self.G, max_iter=100)
            except:
                self.log("  Note: Using approximate betweenness centrality")
                betweenness = nx.betweenness_centrality(self.G, k=100)
            
            centrality_array = np.zeros((num_nodes, 3))
            for i in range(num_nodes):
                node = self.reverse_mapping[i]
                centrality_array[i, 0] = in_centrality[node]
                centrality_array[i, 1] = out_centrality[node]
                centrality_array[i, 2] = betweenness[node]
            
            np.save(self.data_dir / 'centrality_measures.npy', centrality_array)
            self.log(f"✓ Saved centrality_measures.npy")
            
            # ---- Connected Components ----
            self.log("\nAnalyzing connected components...")
            wcc = list(nx.weakly_connected_components(self.G))
            scc = list(nx.strongly_connected_components(self.G))
            
            self.log(f"  - Weakly connected components: {len(wcc)}")
            self.log(f"    Largest WCC size: {len(max(wcc, key=len))}")
            self.log(f"  - Strongly connected components: {len(scc)}")
            self.log(f"    Largest SCC size: {len(max(scc, key=len))}")
            
            # ---- Hub Nodes Analysis ----
            self.log("\nAnalyzing hub nodes (top by degree)...")
            degree_dict = dict(self.G.degree())
            top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for rank, (node_id, degree) in enumerate(top_hubs, 1):
                label = self.G.nodes[node_id].get('label', -1)
                title = self.G.nodes[node_id].get('title', 'N/A')[:50]
                self.log(f"  {rank}. Node {node_id} (label={label}): degree={degree}")
                self.log(f"     Title: {title}...")
            
            self.log("\n✓ Advanced analysis completed successfully\n")
            return True
            
        except Exception as e:
            self.log(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== Task 6: Validation ====================
    def validate_all(self):
        """Validate all generated files and data consistency"""
        self.log("=" * 60)
        self.log("TASK 6: Validation & Quality Check")
        self.log("=" * 60)
        
        validation_passed = True
        issues = []
        
        try:
            # Check 1: nodes.csv and edges.csv consistency
            self.log("Check 1: Data consistency...")
            unique_nodes = set(self.nodes_df['node_id'].unique())
            unique_nodes_in_edges = (
                set(self.edges_df['source'].unique()) | 
                set(self.edges_df['target'].unique())
            )
            
            missing_in_nodes = unique_nodes_in_edges - unique_nodes
            if missing_in_nodes:
                msg = f"  ⚠ {len(missing_in_nodes)} nodes in edges not found in nodes"
                self.log(msg)
                issues.append(msg)
                # Note: This is a data quality issue but not necessarily blocking
            else:
                self.log("  ✓ All nodes in edges are present in nodes.csv")
            
            # Check 2: edge_index.npy exists
            self.log("Check 2: Member 2 Output Files...")
            required_files = ['edge_index.npy']
            for filename in required_files:
                filepath = self.graph_dir / filename
                if filepath.exists():
                    self.log(f"  ✓ {filename} exists")
                else:
                    msg = f"  ✗ {filename} missing"
                    self.log(msg)
                    issues.append(msg)
                    validation_passed = False
            
            # Check 3: Visualization files
            self.log("Check 3: Visualization Files...")
            viz_files = [
                'degree_distribution.png',
                'indegree_outdegree.png',
                'label_distribution.png',
                'graph_visualization.png'
            ]
            for filename in viz_files:
                filepath = self.viz_dir / filename
                if filepath.exists():
                    self.log(f"  ✓ {filename} exists")
                else:
                    self.log(f"  ⚠ {filename} not found")
            
            # Check 4: Graph statistics
            self.log("Check 4: Graph statistics...")
            self.log(f"  - Total nodes: {self.G.number_of_nodes()}")
            self.log(f"  - Total edges: {self.G.number_of_edges()}")
            self.log(f"  - Density: {nx.density(self.G):.4f}")
            
            # Summary
            self.log("\n" + "=" * 60)
            if validation_passed:
                self.log("✓ VALIDATION PASSED")
            else:
                self.log("✗ VALIDATION FAILED - Issues found:")
                for issue in issues:
                    self.log(f"  {issue}")
            self.log("=" * 60 + "\n")
            
            return validation_passed
            
        except Exception as e:
            self.log(f"✗ ERROR during validation: {e}")
            return False
    
    # ==================== Main Execution ====================
    def run_all(self, task='build-all'):
        """Execute specified task or all tasks for Member 2"""
        start_time = datetime.now()
        
        try:
            if task == 'build-graph':
                self.load_data()
                self.build_graph()
            
            elif task == 'create-edge-index':
                self.load_data()
                self.build_graph()
                self.create_edge_index()
            
            elif task == 'visualize':
                self.load_data()
                self.build_graph()
                self.visualize()
            
            elif task == 'analyze-advanced':
                self.load_data()
                self.build_graph()
                self.create_edge_index()
                self.analyze_advanced()
            
            elif task == 'build-all':
                # Member 2 main task: create edge_index + visualizations
                self.load_data()
                self.build_graph()
                self.create_edge_index()
                self.visualize()
                self.validate_all()
            
            else:
                self.log(f"Unknown task: {task}")
                return False
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.log(f"\n{'='*60}")
            self.log(f"✓ EXECUTION COMPLETED in {elapsed:.1f}s")
            self.log(f"{'='*60}\n")
            
            # Note: Logs not saved to disk (console only)
            return True
            
        except Exception as e:
            self.log(f"\n✗ EXECUTION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Graph Construction for Fake News Detection'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='build-all',
        choices=[
            'build-graph',
            'create-edge-index',
            'visualize',
            'analyze-advanced',
            'build-all'
        ],
        help='Task to execute'
    )
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create builder and run
    builder = GraphBuilder(data_dir=args.data_dir, output_dir=args.output_dir)
    success = builder.run_all(task=args.task)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
