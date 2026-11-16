"""Generate the fictional, redistributable maintenance PDF used for the demo."""

from pathlib import Path

import pymupdf


def create_demo_pdf(path: Path):
    with pymupdf.open() as illustration:
        panel = illustration.new_page(width=440, height=150)
        panel.draw_rect(panel.rect, color=(0.2, 0.3, 0.4), fill=(0.94, 0.96, 0.98))
        panel.insert_text((25, 30), "MX-10 CONTROL PANEL", fontsize=18)
        for x, label, color in [
            (80, "POWER", (0.1, 0.65, 0.3)),
            (220, "STATUS", (1, 0.55, 0.05)),
            (360, "STOP", (0.85, 0.15, 0.15)),
        ]:
            panel.draw_circle((x, 75), 15, color=(0.15, 0.2, 0.25), fill=color)
            panel.insert_text((x - 25, 120), label, fontsize=12)
        image = panel.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).tobytes("png")

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 55), "MX-10 - Maintenance training manual", fontsize=19)
        page.insert_textbox(pymupdf.Rect(50, 80, 545, 180), (
            "Fictional equipment for a software demonstration. This is not a real maintenance procedure.\n"
            "The MX-10 is a training pump. A steady orange STATUS light means the filter needs inspection "
            "at the next authorized stop. The indicator alone does not identify the root cause.\n"
            "Only an authorized technician may inspect equipment using the applicable site procedures."
        ), fontsize=11)
        page.insert_text((50, 205), "Table 1. Indicator meanings", fontsize=13)
        rows = [
            ["Indicator", "Meaning"],
            ["POWER - green", "Electrical supply present"],
            ["STATUS - orange", "Filter inspection required"],
            ["STOP - red", "Follow the site's stop procedure"],
        ]
        xs, top, height = [50, 215, 545], 225, 38
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                rectangle = pymupdf.Rect(xs[j], top + i * height, xs[j + 1], top + (i + 1) * height)
                page.draw_rect(rectangle, color=(0, 0, 0), width=0.6)
                page.insert_text((xs[j] + 8, top + i * height + 23), value, fontsize=11)
        page = document.new_page()
        page.insert_text((50, 55), "MX-10 - Control panel", fontsize=19)
        page.insert_image(pymupdf.Rect(50, 95, 545, 264), stream=image)
        page.insert_textbox(pymupdf.Rect(50, 290, 545, 390), (
            "Figure 1. MX-10 control panel. The visible labels are POWER, STATUS and STOP.\n"
            "Use Table 1 on page 1 for the indicator meanings.\n"
            "Record the equipment identifier and the observed light before preparing an intervention."
        ), fontsize=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(path)


if __name__ == "__main__":
    output = Path(__file__).with_name("maintenance_demo.pdf")
    create_demo_pdf(output)
    print(output)
