#!/usr/bin/env python3
"""Build the official submission ZIP archive for Stud.IP: 1837734.zip containing exactly poster.pdf and appendix.pdf."""
import os
import sys
import zipfile

def main():
    poster_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(poster_dir)

    student_id = "1837734"
    zip_name = f"{student_id}.zip"
    required_files = ["poster.pdf", "appendix.pdf"]

    print("=" * 80)
    print(f" BUILDING SUBMISSION ARCHIVE: {zip_name}")
    print("=" * 80)

    for f in required_files:
        if not os.path.exists(f):
            print(f"ERROR: Missing required file {f}. Please compile it first.")
            sys.exit(1)

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for name in required_files:
            z.write(name, arcname=name)
            print(f"  Added: {name} ({os.path.getsize(name):,} bytes)")

    # Verify zip content
    with zipfile.ZipFile(zip_name, "r") as z:
        contents = z.namelist()
        print(f"\nVerification of {zip_name}:")
        print(f"  Files inside archive: {contents}")
        assert set(contents) == set(required_files), f"Archive must contain exactly {required_files}, got {contents}"
        assert len(contents) == 2, f"Archive must contain exactly 2 files, got {len(contents)}"

    print("\n" + "=" * 80)
    print(f" SUBMISSION READY: {zip_name} created successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
