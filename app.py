# app.py

import gradio as gr
from src.inference.infer import run_inference
import tempfile
import traceback


def generate_layout(sketch, green_intensity, building_density, use_canny):
    if sketch is None:
        return None, {}

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()

        sketch.save(tmp_path)

        result, metrics = run_inference(
            tmp_path,
            green_intensity,
            building_density,
            use_canny=use_canny
        )

        return result, metrics

    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()
        return None, {}


with gr.Blocks(title="SustainableUrbanPix2Pix") as demo:
    gr.Markdown(
        "# 🌆 Sustainable Urban Layout Generator\n"
        "Convert sketches → photorealistic sustainable cities\n"
        "**Optimized for GTX 1050 Ti**"
    )

    with gr.Tabs():

        # ---------------------------
        # 🎨 GENERATOR TAB
        # ---------------------------
        with gr.Tab("🎨 Generator"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Sketch")
                    use_canny = gr.Checkbox(label="Apply Canny", value=True)
                    green_slider = gr.Slider(0, 1, 0.65, label="Green Intensity")
                    density_slider = gr.Slider(0, 1, 0.75, label="Building Density")

                with gr.Column():
                    output_image = gr.Image(label="Generated Output")

        # ---------------------------
        # 🎯 METRICS TAB
        # ---------------------------
        with gr.Tab("🎯 Metrics Dashboard"):
            metrics_output = gr.JSON(label="Generation Metrics")

    btn = gr.Button("Generate", variant="primary")

    btn.click(
        generate_layout,
        inputs=[input_image, green_slider, density_slider, use_canny],
        outputs=[output_image, metrics_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)