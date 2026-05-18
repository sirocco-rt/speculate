"""
Speculate Grid Compression Utility
Compress/decompress .spec files using LZMA/XZ for maximum compression ratio
"""
import os
import lzma
from tqdm import tqdm

def compress_spec_files(directory="."):
    """Compress all .spec files in directory using LZMA"""
    spec_files = [f for f in os.listdir(directory) if f.endswith(".spec")]
    
    if not spec_files:
        print("No .spec files found")
        return
    
    # Create output directory with _xz suffix
    # Get the directory name (handle both relative and absolute paths)
    dir_name = os.path.basename(os.path.abspath(directory))
    parent_dir = os.path.dirname(os.path.abspath(directory))
    output_dir = os.path.join(parent_dir, dir_name + "_xz")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"Compressing {len(spec_files)} .spec files with LZMA (level 9)...")
    total_original = 0
    total_compressed = 0
    
    for spec_file in tqdm(spec_files, desc="Compressing"):
        filepath = os.path.join(directory, spec_file)
        compressed_path = os.path.join(output_dir, spec_file + '.xz')
        
        # Read and compress
        with open(filepath, 'rb') as f_in:
            data = f_in.read()
            total_original += len(data)
        
        with lzma.open(compressed_path, 'wb', preset=9) as f_out:
            f_out.write(data)
        
        total_compressed += os.path.getsize(compressed_path)
    
    print(f"\n✅ Compression complete!")
    print(f"   Original:   {total_original / 1024 / 1024:.2f} MB")
    print(f"   Compressed: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"   Ratio:      {total_original / total_compressed:.2f}x")
    print(f"   Saved:      {(total_original - total_compressed) / 1024 / 1024:.2f} MB")
    print(f"   Location:   {output_dir}")

def decompress_spec_files(directory="."):
    """Decompress all .spec.xz files in directory"""
    compressed_files = [f for f in os.listdir(directory) if f.endswith(".spec.xz")]
    
    if not compressed_files:
        print("No .spec.xz files found")
        return
    
    # Create output directory by removing _xz suffix if present
    dir_name = os.path.basename(os.path.abspath(directory))
    parent_dir = os.path.dirname(os.path.abspath(directory))
    
    if dir_name.endswith("_xz"):
        output_dir_name = dir_name[:-3]  # Remove _xz suffix
    else:
        output_dir_name = dir_name + "_decompressed"
    
    output_dir = os.path.join(parent_dir, output_dir_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"Decompressing {len(compressed_files)} .spec.xz files...")
    
    for compressed_file in tqdm(compressed_files, desc="Decompressing"):
        filepath = os.path.join(directory, compressed_file)
        output_filename = compressed_file[:-3]  # Remove .xz extension
        output_path = os.path.join(output_dir, output_filename)
        
        with lzma.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    
    print(f"\n✅ Decompression complete! {len(compressed_files)} files restored")
    print(f"   Location: {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Compress:   python compress_grid.py compress [directory]")
        print("  Decompress: python compress_grid.py decompress [directory]")
        print("\nExamples:")
        print("  python compress_grid.py compress")
        print("  python compress_grid.py compress ../CV_release_grid_spec")
        print("  python compress_grid.py decompress foldername/")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    # Get directory from command line or use current directory
    directory = "."
    if len(sys.argv) >= 3:
        directory = sys.argv[2]
        # Ensure directory doesn't end with slash for consistency (os.path handles both)
        directory = directory.rstrip('/').rstrip('\\')
    
    # Verify directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    print(f"Working in directory: {directory}\n")
    
    if action == "compress":
        compress_spec_files(directory)
    elif action == "decompress":
        decompress_spec_files(directory)
    else:
        print(f"Unknown action: {action}")
        print("Use 'compress' or 'decompress'")