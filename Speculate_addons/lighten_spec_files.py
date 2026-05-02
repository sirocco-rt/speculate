"""Reduce native Sirocco `.spec` files to the columns Speculate needs."""

import os
import sys

import astropy.table as at
import tqdm


DEFAULT_COLUMNS_TO_REMOVE = (
    "Created",
    "WCreated",
    "Emitted",
    "CenSrc",
    "Disk",
    "Wind",
    "HitSurf",
    "Scattered",
)


def _default_output_dir(directory):
    """Return the sibling default output directory for reduced `.spec` files."""
    dir_name = os.path.basename(os.path.abspath(directory))
    parent_dir = os.path.dirname(os.path.abspath(directory))
    return os.path.join(parent_dir, dir_name + "_reduced")


def reduce_spec_file(
    spec_path,
    output_path,
    columns_to_remove=DEFAULT_COLUMNS_TO_REMOVE,
):
    """Reduce one .spec file by removing diagnostic columns when present."""
    with open(spec_path, "r") as f:
        lines = f.readlines()

    data = at.Table.read(lines, format="ascii")
    removable = [name for name in columns_to_remove if name in data.colnames]
    if removable:
        data.remove_columns(removable)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data.write(
        output_path,
        format="ascii.fixed_width",
        delimiter=" ",
        bookend=False,
        overwrite=True,
    )
    return output_path


def reduce_spec_files(
    directory=".",
    output_dir=None,
    columns_to_remove=DEFAULT_COLUMNS_TO_REMOVE,
    show_progress=True,
    strict=False,
):
    """Reduce all .spec files in a directory and return written file paths."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    output_dir = output_dir or _default_output_dir(directory)
    os.makedirs(output_dir, exist_ok=True)

    run_files = sorted(
        name for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name)) and name.endswith(".spec")
    )
    if not run_files:
        return []

    iterator = run_files
    if show_progress:
        iterator = tqdm.tqdm(run_files, desc="Reducing files")

    written = []
    for file_name in iterator:
        spec_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(output_dir, file_name)
        try:
            written.append(
                reduce_spec_file(
                    spec_path,
                    new_file_path,
                    columns_to_remove=columns_to_remove,
                )
            )
        except Exception as exc:
            if strict:
                raise
            print(f"\nError processing {file_name}: {exc}")

    return written


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Reduce: python lighten_spec_files.py reduce [directory] [output_directory]")
        print("\nExamples:")
        print("  python lighten_spec_files.py reduce")
        print("  python lighten_spec_files.py reduce ../CV_release_grid_spec")
        print("  python lighten_spec_files.py reduce folder_name/")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action != "reduce":
        print(f"Unknown action: {action}")
        print("Use 'reduce' to remove unnecessary columns from .spec files")
        sys.exit(1)
    
    # Get directory from command line or use current directory
    directory = "."
    if len(sys.argv) >= 3:
        directory = sys.argv[2]
        # Ensure directory doesn't end with slash for consistency
        directory = directory.rstrip('/').rstrip('\\')
    
    output_dir = None
    if len(sys.argv) >= 4:
        output_dir = sys.argv[3].rstrip('/').rstrip('\\')

    try:
        written = reduce_spec_files(directory, output_dir=output_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    final_output_dir = output_dir or _default_output_dir(directory)
    print(f"Input directory:  {directory}")
    print(f"Output directory: {final_output_dir}\n")
    if not written:
        print("No .spec files found")
        sys.exit(0)
    print("Reduction complete")
    print(f"   Processed: {len(written)} files")
    print(f"   Location:  {final_output_dir}")

