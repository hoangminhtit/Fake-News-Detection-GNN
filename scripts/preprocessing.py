#!/usr/bin/env python3
"""
Preprocessing Module for Fake News Detection Project
Data Preprocessing
Converts raw data (true.csv, fake.csv) → processed data (nodes.csv, edges.csv)
"""

import pandas as pd
import numpy as np
import string
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime

# Determine project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


class DataPreprocessor:
    """Main class for data preprocessing"""
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed', verbose=True):
        """Initialize preprocessor"""
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        
        # Data structures
        self.df = None
        self.df_sample = None
        self.y_data = None
        self.tfidf_features = None
        self.adjacency_sparse = None
        
    def log(self, msg):
        """Print message"""
        if self.verbose:
            print(msg)
    
    # ==================== Step 1: Load Data ====================
    def load_data(self, sample_size=-1, mode='split', input_file=None):
        """Load data - support 2 modes:
        - 'split': Load from true.csv + fake.csv (2 files)
        - 'single': Load from single CSV file with 'label' column
        """
        self.log("=" * 60)
        self.log("STEP 1: Loading Raw Data")
        self.log("=" * 60)
        
        try:
            if mode == 'split':
                # Mode 1: Load 2 separate files
                self.log("Mode: Split files (true.csv + fake.csv)")
                
                fake_path = self.raw_dir / 'fake.csv'
                true_path = self.raw_dir / 'true.csv'
                
                if not fake_path.exists() or not true_path.exists():
                    self.log(f"  [ERROR] Split mode files not found")
                    
                    # Auto-detect: Look for single CSV file with label column
                    csv_files = list(self.raw_dir.glob('*.csv'))
                    self.log(f"  Available files in {self.raw_dir}:")
                    for file in csv_files:
                        self.log(f"    - {file.name}")
                    
                    if len(csv_files) == 1:
                        self.log(f"\n  [AUTO-SWITCH] Switching to single-file mode...")
                        single_file = csv_files[0]
                        try:
                            test_df = pd.read_csv(single_file)
                            if 'label' in test_df.columns:
                                self.log(f"  [OK] Single file with 'label' column found: {single_file.name}")
                                self.log(f"  Retrying in single-file mode...\n")
                                return self.load_data(sample_size=sample_size, mode='single', input_file=str(single_file))
                            else:
                                raise ValueError("'label' column not found in the single CSV file")
                        except Exception as e:
                            self.log(f"  [ERROR] Could not auto-switch: {e}")
                            raise
                    else:
                        raise FileNotFoundError(f"Files {fake_path} and {true_path} not found. Use 'single' mode for single-file datasets or provide both split files.")
                
                self.log(f"Loading {fake_path}...")
                fake_df = pd.read_csv(fake_path)
                fake_df['label'] = 1  # 1 for fake
                self.log(f"  [OK] Fake news: {len(fake_df)} articles")
                
                self.log(f"Loading {true_path}...")
                true_df = pd.read_csv(true_path)
                true_df['label'] = 0  # 0 for true
                self.log(f"  [OK] True news: {len(true_df)} articles")
                
                # Combine datasets
                self.df = pd.concat([fake_df, true_df], ignore_index=True)
            
            elif mode == 'single':
                # Mode 2: Load single file with label column
                self.log("Mode: Single file with 'label' column")
                
                if input_file is None:
                    raise ValueError("input_file must be specified for 'single' mode")
                
                input_path = Path(input_file)
                if not input_path.exists():
                    raise FileNotFoundError(f"File not found: {input_path}")
                
                self.log(f"Loading {input_path}...")
                self.df = pd.read_csv(input_path)
                
                # Check for label column
                if 'label' not in self.df.columns:
                    raise ValueError("'label' column not found in input file")
                
                self.log(f"  [OK] Loaded {len(self.df)} articles")
                
                # Show label distribution
                label_counts = self.df['label'].value_counts()
                self.log(f"  [OK] Label distribution:")
                for label, count in label_counts.items():
                    label_name = "Fake" if label == 1 else "Real"
                    self.log(f"    - {label_name} ({label}): {count} articles")
            
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'split' or 'single'")
            
            self.log(f"\n[OK] Combined dataset shape: {self.df.shape}")
            
            # Sample data: use -1 to get all data, else sample N records
            if sample_size < 0:
                # -1 means use all data
                self.df_sample = self.df.reset_index(drop=True)
                self.log(f"[OK] Using all {len(self.df)} articles for processing")
            else:
                self.df_sample = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                self.log(f"[OK] Sampled {sample_size} articles for processing")
            
            # Check columns
            self.log(f"\nAvailable columns: {list(self.df_sample.columns)}")
            
            self.log("\n[OK] Data loading completed\n")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] ERROR: {e}")
            return False
    
    # ==================== Step 2: Text Preprocessing ====================
    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = str(text).lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def preprocess_texts(self):
        """Apply text preprocessing to all samples"""
        self.log("=" * 60)
        self.log("STEP 2: Text Preprocessing")
        self.log("=" * 60)
        
        try:
            # Preprocess title and text columns
            if 'title' in self.df_sample.columns:
                self.log("Processing 'title' column...")
                self.df_sample['title'] = self.df_sample['title'].astype(str).apply(self.preprocess_text)
                self.log("  [OK] Title preprocessing complete")
            
            if 'text' in self.df_sample.columns:
                self.log("Processing 'text' column...")
                self.df_sample['text'] = self.df_sample['text'].astype(str).apply(self.preprocess_text)
                self.log("  [OK] Text preprocessing complete")
            
            # Combine title and text into content
            content_parts = []
            if 'title' in self.df_sample.columns:
                content_parts.append(self.df_sample['title'])
            if 'text' in self.df_sample.columns:
                content_parts.append(self.df_sample['text'])
            
            if content_parts:
                self.df_sample['content'] = ' '.join([str(part) for part in content_parts])
                # Alternative: use Series concatenation
                self.df_sample['content'] = pd.concat(content_parts, axis=1).fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
                self.log("  [OK] Combined title + text into 'content'")
            
            self.log("\n[OK] Text preprocessing completed\n")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] ERROR: {e}")
            return False
    
    # ==================== Step 3: Feature Extraction ====================
    def extract_features(self, max_features=2000):
        """Extract TF-IDF features from content"""
        self.log("=" * 60)
        self.log("STEP 3: Feature Extraction (TF-IDF)")
        self.log("=" * 60)
        
        try:
            if 'content' not in self.df_sample.columns:
                raise ValueError("'content' column not found. Run preprocess_texts() first.")
            
            # Create TF-IDF vectorizer
            self.log(f"Creating TF-IDF vectorizer (max_features={max_features})...")
            vectorizer = TfidfVectorizer(max_features=max_features, strip_accents='unicode', analyzer='word')
            tfidf_matrix = vectorizer.fit_transform(self.df_sample['content'])
            self.tfidf_features = tfidf_matrix.toarray()
            self.log(f"  [OK] TF-IDF matrix shape: {self.tfidf_features.shape}")
            
            # Normalize features
            self.log("Normalizing features...")
            scaler = StandardScaler()
            self.tfidf_features = scaler.fit_transform(self.tfidf_features)
            self.log("  [OK] Features normalized (mean=0, std=1)")
            
            self.log("\n[OK] Feature extraction completed\n")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] ERROR: {e}")
            return False
    
    # ==================== Step 4: Build KNN Graph ====================
    def build_knn_graph(self, k=5):
        """Build KNN graph from TF-IDF features"""
        self.log("=" * 60)
        self.log("STEP 4: Building KNN Graph")
        self.log("=" * 60)
        
        try:
            if self.tfidf_features is None:
                raise ValueError("Features not extracted. Run extract_features() first.")
            
            self.log(f"Finding {k} nearest neighbors (cosine similarity)...")
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='cosine').fit(self.tfidf_features)
            distances, indices = nbrs.kneighbors(self.tfidf_features)
            self.log(f"  [OK] Neighbors found")
            
            # Build adjacency matrix
            self.log("Building adjacency matrix...")
            rows = []
            cols = []
            
            for i in range(self.tfidf_features.shape[0]):
                for j in range(1, k+1):  # Skip first neighbor (itself)
                    neighbor = indices[i][j]
                    rows.append(i)
                    cols.append(neighbor)
            
            data = np.ones(len(rows))
            adjacency_sparse = csr_matrix((data, (rows, cols)), 
                                         shape=(self.tfidf_features.shape[0], self.tfidf_features.shape[0]))
            
            # Symmetrize for undirected graph
            self.log("Symmetrizing graph (undirected edges)...")
            self.adjacency_sparse = adjacency_sparse.maximum(adjacency_sparse.transpose())
            
            # Statistics
            num_nodes = self.adjacency_sparse.shape[0]
            num_edges = self.adjacency_sparse.count_nonzero()
            
            self.log(f"\n[OK] Graph statistics:")
            self.log(f"  - Nodes: {num_nodes}")
            self.log(f"  - Edges (undirected): {num_edges}")
            self.log(f"  - Density: {num_edges / (num_nodes * (num_nodes - 1)) * 2:.4f}")
            
            self.log("\n[OK] KNN graph building completed\n")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] ERROR: {e}")
            return False
    
    # ==================== Step 5: Create Nodes & Edges CSV ====================
    def export_graph_data(self):
        """Export graph data to CSV files"""
        self.log("=" * 60)
        self.log("STEP 5: Exporting Graph Data")
        self.log("=" * 60)
        
        try:
            if self.adjacency_sparse is None:
                raise ValueError("Graph not built. Run build_knn_graph() first.")
            
            # Extract labels
            self.y_data = self.df_sample['label'].values
            
            # ---- Create nodes.csv ----
            self.log("Creating nodes.csv...")
            nodes_df = pd.DataFrame({
                'node_id': np.arange(len(self.df_sample)),
                'label': self.y_data
            })
            
            # Add title if available
            if 'title' in self.df_sample.columns:
                nodes_df['title'] = self.df_sample['title'].values
            
            nodes_path = self.processed_dir / 'nodes.csv'
            nodes_df.to_csv(nodes_path, index=False)
            self.log(f"  [OK] Saved {nodes_path} ({len(nodes_df)} nodes)")
            
            # ---- Create edges.csv ----
            self.log("Creating edges.csv...")
            adjacency_coo = self.adjacency_sparse.tocoo()
            
            edges_df = pd.DataFrame({
                'source': adjacency_coo.row,
                'target': adjacency_coo.col,
                'weight': adjacency_coo.data
            })
            
            # Keep only unique undirected edges (source < target)
            edges_df = edges_df[edges_df['source'] < edges_df['target']].reset_index(drop=True)
            
            edges_path = self.processed_dir / 'edges.csv'
            edges_df.to_csv(edges_path, index=False)
            self.log(f"  [OK] Saved {edges_path} ({len(edges_df)} edges)")
            
            # Print summary
            self.log(f"\n[OK] Data export summary:")
            self.log(f"  - Nodes: {len(nodes_df)}")
            self.log(f"  - Edges: {len(edges_df)}")
            
            fake_count = (nodes_df['label'] == 1).sum()
            real_count = (nodes_df['label'] == 0).sum()
            self.log(f"  - Fake news: {fake_count} ({fake_count/len(nodes_df)*100:.1f}%)")
            self.log(f"  - Real news: {real_count} ({real_count/len(nodes_df)*100:.1f}%)")
            
            self.log("\n[OK] Data export completed\n")
            return True
            
        except Exception as e:
            self.log(f"[ERROR] ERROR: {e}")
            return False
    
    # ==================== Main Execution ====================
    def run_pipeline(self, sample_size=-1, mode='split', input_file=None):
        """Execute full preprocessing pipeline
        
        Args:
            sample_size: Number of samples to process
            mode: 'split' (true.csv + fake.csv) or 'single' (file with label column)
            input_file: Required for 'single' mode
        """
        start_time = datetime.now()
        
        self.log(f"\n{'='*60}")
        self.log(f"Preprocessing Pipeline Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Mode: {mode} | Sample size: {sample_size}")
        self.log(f"{'='*60}\n")
        
        try:
            # Step 1: Load
            if not self.load_data(sample_size=sample_size, mode=mode, input_file=input_file):
                return False
            
            # Step 2: Preprocess text
            if not self.preprocess_texts():
                return False
            
            # Step 3: Extract features
            if not self.extract_features():
                return False
            
            # Step 4: Build graph
            if not self.build_knn_graph(k=5):
                return False
            
            # Step 5: Export
            if not self.export_graph_data():
                return False
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.log(f"{'='*60}")
            self.log(f"[OK] PREPROCESSING COMPLETED in {elapsed:.1f}s")
            self.log(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            self.log(f"\n[ERROR] PREPROCESSING FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
