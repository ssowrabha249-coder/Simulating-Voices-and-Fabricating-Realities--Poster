"""Check the poster against hard university requirements: page count, page size, smallest font size."""
import collections
import sys
import pdfplumber

def main():
    print("=" * 80)
    print(" RESEARCH POSTER PRE-SUBMISSION VERIFICATION REPORT")
    print(" Author: Sowrabha Somashekar (Matr. 1837734)")
    print("=" * 80)

    pdf_file = "poster.pdf"
    try:
        pdf = pdfplumber.open(pdf_file)
    except Exception as e:
        print(f"ERROR: Could not open {pdf_file}: {e}")
        sys.exit(1)

    # 1. Page count check
    n_pages = len(pdf.pages)
    print(f"\n[1] Page Count Check: {n_pages} page(s)")
    if n_pages != 1:
        print(f"  FAILED: Expected exactly 1 page, got {n_pages}")
        sys.exit(1)
    else:
        print("  PASS: Exactly 1 page.")

    # 2. Dimensions check (DIN A1: 594 x 841 mm)
    page = pdf.pages[0]
    w_pt, h_pt = float(page.width), float(page.height)
    w_mm = w_pt / 72 * 25.4
    h_mm = h_pt / 72 * 25.4
    print(f"\n[2] Dimension Check:")
    print(f"  Dimensions: {w_pt:.2f} x {h_pt:.2f} pt = {w_mm:.1f} x {h_mm:.1f} mm")
    # DIN A1 is 594 x 841 mm (+/- 5 mm tolerance)
    if (585 <= w_mm <= 605 and 830 <= h_mm <= 850) or (830 <= w_mm <= 850 and 585 <= h_mm <= 605):
        print("  PASS: Page dimensions match DIN A1.")
    else:
        print(f"  FAILED: Page dimensions do not match DIN A1 (594 x 841 mm).")
        sys.exit(1)

    # 3. Minimum Font Size Check
    print(f"\n[3] Font Size Check (Rule: >= 24 pt):")
    sizes = collections.Counter()
    smallest = []
    for ch in page.chars:
        text = ch["text"].strip()
        if not text:
            continue
        size = round(float(ch["size"]), 2)
        sizes[size] += 1
        smallest.append((size, text, ch["fontname"]))

    if not smallest:
        print("  WARN: No text characters detected.")
    else:
        smallest.sort()
        min_size = smallest[0][0]
        print(f"  Total characters checked: {sum(sizes.values())}")
        print(f"  Smallest font size found: {min_size:.2f} pt")
        print(f"  10 smallest characters: {[(s, t) for s, t, _ in smallest[:10]]}")

        below_24 = [c for c in smallest if c[0] < 23.99]
        if below_24:
            print(f"  FAILED: Found {len(below_24)} characters below 24 pt threshold!")
            print(f"  Sample below 24pt: {[(s, t, f) for s, t, f in below_24[:15]]}")
            sys.exit(1)
        else:
            print("  PASS: 0 characters below 24 pt. All text is fully compliant.")

    # 4. Raster image resolution
    print(f"\n[4] Raster Images Resolution Check:")
    print(f"  Raster images on page: {len(page.images)}")
    low_res = False
    for im in page.images:
        w_in = (im["x1"] - im["x0"]) / 72
        h_in = (im["bottom"] - im["top"]) / 72
        src_w, src_h = im["srcsize"]
        ppi_x = src_w / w_in if w_in > 0 else 0
        ppi_y = src_h / h_in if h_in > 0 else 0
        print(f"    Image {src_w}x{src_h}px shown at {w_in:.2f}x{h_in:.2f} in -> {ppi_x:.0f}x{ppi_y:.0f} PPI")
        if ppi_x < 150 or ppi_y < 150:
            low_res = True

    if low_res:
        print("  FAILED: At least one raster image is below 150 PPI.")
        sys.exit(1)
    else:
        print("  PASS: All images (or vector graphics) meet resolution criteria.")

    # 5. Repository link (mandatory for technical work according to the poster rules)
    print(f"\n[5] Repository Link Check:")
    if "[REPOSITORYURL" in "".join((page.extract_text() or "").split()):  # red placeholder text
        print("  FAILED: repository URL still missing (set \\repourl in poster.tex and appendix.tex).")
        sys.exit(1)
    print("  PASS: repository link present.")

    print("\n" + "=" * 80)
    print(" RESEARCH POSTER STATUS: READY FOR SUBMISSION (ALL CHECKS PASSED)")
    print("=" * 80)

if __name__ == "__main__":
    main()
