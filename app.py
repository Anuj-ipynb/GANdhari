# filename: app.py

import gradio as gr
from src.inference.infer import run_inference
import tempfile
import os
import traceback


def generate_layout(sketch, green_intensity, building_density, use_canny):
    if sketch is None:
        return None

    try:
        # ✅ Create temp file safely (Windows compatible)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()  # 🔥 VERY IMPORTANT (releases file lock)

        # Save sketch
        sketch.save(tmp_path)

        # Run inference
        result = run_inference(
            tmp_path,
            green_intensity,
            building_density,
            use_canny=use_canny
        )

        return result

    except Exception as e:
        print("\n❌ ERROR IN APP:")
        traceback.print_exc()
        return None


with gr.Blocks(title="SustainableUrbanPix2Pix") as demo:
    gr.Markdown(
        "# 🌆 Sustainable Urban Layout Generator\n"
        "Convert your hand-drawn city plans into photorealistic sustainable designs.\n"
        "**Optimized for GTX 1050 Ti**"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Raw Hand-Drawn Plan (JPG/PNG)"
            )

            use_canny = gr.Checkbox(
                label="Apply Canny Edge Cleaning",
                value=True
            )

            green_slider = gr.Slider(
                0.0, 1.0,
                value=0.65,
                step=0.05,
                label="Green Intensity (Parks & Trees)"
            )

            density_slider = gr.Slider(
                0.0, 1.0,
                value=0.75,
                step=0.05,
                label="Building Density (Lower = More Open Space)"
            )

        with gr.Column():
            output_image = gr.Image(
                label="Generated Photorealistic Sustainable Layout"
            )

    btn = gr.Button(
        "Generate Sustainable Urban Layout",
        variant="primary"
    )

    btn.click(
        generate_layout,
        inputs=[input_image, green_slider, density_slider, use_canny],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)