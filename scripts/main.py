import sys
from pathlib import Path

# Determine project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Handle imports for both direct script execution and module import

from preprocessing import DataPreprocessor
from graph_construction import GraphBuilder
from visualization import GraphVisualizer


def main():
    option = sys.argv[1].lower() if len(sys.argv) > 1 else 'full'
    
    # Step 1: Preprocessing
    if option in ['full', 'preprocess']:
        print("\n[1/3] Preprocessing...")
        raw_dir = PROJECT_ROOT / 'data' / 'raw'
        processed_dir = PROJECT_ROOT / 'data' / 'processed'
        
        preprocessor = DataPreprocessor(
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            verbose=True
        )
        
        # Auto-detect data files
        csv_files = list(raw_dir.glob('*.csv'))
        
        if len(csv_files) == 2:
            # Two files found - use split mode
            mode = 'split'
            preprocessor.run_pipeline(sample_size=3000, mode=mode)
        elif len(csv_files) == 1:
            # Single file found - use single mode
            mode = 'single'
            input_file = str(csv_files[0])
            preprocessor.run_pipeline(sample_size=3000, mode=mode, input_file=input_file)
        else:
            print(f"✗ Error: No data files found in {raw_dir}")
            print("  Please provide either:")
            print("    - true.csv and fake.csv (split mode)")
            print("    - OR one CSV file with 'label' column (single mode)")
            return False
    
    # Step 2: Graph Construction
    if option in ['full', 'graph']:
        print("\n[2/3] Graph Construction...")
        data_dir = PROJECT_ROOT / 'data'
        processed = data_dir / 'processed'
        if not (processed / 'nodes.csv').exists():
            print("✗ Error: Run preprocessing first")
            return False
        
        builder = GraphBuilder(data_dir=str(data_dir), verbose=True)
        if not (builder.load_data() and builder.build_graph() and builder.create_edge_index()):
            return False
    
    # Step 3: Visualization
    if option in ['full', 'visualize']:
        print("\n[3/3] Visualization...")
        data_dir = PROJECT_ROOT / 'data'
        graph_dir = data_dir / 'graph'
        if not (graph_dir / 'edge_index.npy').exists():
            print("✗ Error: Build graph first")
            return False
        
        builder = GraphBuilder(data_dir=str(data_dir), verbose=False)
        builder.load_data()
        builder.build_graph()
        
        viz_dir = PROJECT_ROOT / 'output' / 'visualizations'
        viz = GraphVisualizer(builder.G, builder.nodes_df, viz_dir=str(viz_dir), verbose=True)
        viz.visualize_all()
    
    if option not in ['full', 'preprocess', 'graph', 'visualize']:
        print(f"✗ Invalid option: {option}")
        return False
    
    print("\n✓ Done\n")
    return True


if __name__ == '__main__':
    success = main()
