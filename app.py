# app.py

import gradio as gr
from src.inference.infer import run_inference
import tempfile
import traceback
import json
import plotly.graph_objects as go
import os
import numpy as np

# -----------------------------
# 🧠 Enhanced Insight Engine
# -----------------------------
def interpret(metrics):
    if not metrics or "error" in metrics:
        return "Error generating metrics."

    g = float(metrics.get("green_coverage", 0))
    d = float(metrics.get("building_density", 0))
    r = float(metrics.get("road_coverage", 0))
    score = float(metrics.get("sustainability_score", 50))
    conn = float(metrics.get("road_connectivity", 0.5))

    msg = []

    if g < 25:
        msg.append("⚠️ CRITICAL: Very low green coverage (<25%). High Urban Heat Island risk in Bengaluru. Add pocket parks and green corridors.")
    elif g < 35:
        msg.append("⚠️ WARNING: Moderate green space. Recommend tree-lined streets and rooftop gardens.")

    if d > 45:
        msg.append("⚠️ WARNING: High building density (>45%). Risk of overcrowding. Consider vertical mixed-use.")

    if r < 12:
        msg.append("⚠️ CRITICAL: Poor road coverage (<12%). Improve grid connectivity.")

    if conn > 0.8:
        msg.append("⚠️ NOTE: Fragmented road network. Reduce dead-ends for better walkability.")

    if score >= 80:
        msg.append("✅ EXCELLENT: Highly sustainable layout.")
    elif score >= 65:
        msg.append("✅ GOOD: Solid foundation. Minor green/road improvements recommended.")
    else:
        msg.append("⚠️ NEEDS IMPROVEMENT: Focus on increasing green coverage and connectivity.")

    return "\n".join(msg)


# -----------------------------
# 🚀 Main Inference Wrapper (Fixed JSON Serialization)
# -----------------------------
def generate_layout(sketch, green_intensity, building_density, use_canny):
    if sketch is None:
        return None, {}, "", None, None

    try:
        # Save temp sketch
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()
        sketch.save(tmp_path)

        # Run inference
        result_img, metrics = run_inference(
            tmp_path,
            green_intensity,
            building_density,
            use_canny=use_canny
        )

        if result_img is None or "error" in metrics:
            return None, metrics, "Inference failed. Check console.", None, None

        insight = interpret(metrics)

        # Create Radar Chart
        fig = None
        try:
            cats = ["Green", "Road", "1-Density", "Connectivity", "OSR"]
            vals = [
                float(metrics.get("green_coverage", 0)),
                float(metrics.get("road_coverage", 0)),
                100 - float(metrics.get("building_density", 0)),
                100 - float(metrics.get("road_connectivity", 0)) * 10,
                float(metrics.get("osr_proxy", 0)) * 100
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=420,
                title="Sustainability Radar Chart"
            )
        except Exception as radar_err:
            print("Radar chart warning:", radar_err)

        # === FIXED: Make metrics JSON serializable ===
        os.makedirs("outputs/results", exist_ok=True)
        json_path = "outputs/results/sustainability_results.json"

        # Convert all numpy types to Python native types
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable_metrics[k] = int(v)
            else:
                serializable_metrics[k] = v

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2)

        return result_img, serializable_metrics, insight, fig, json_path

    except Exception as e:
        print("\n❌ ERROR in generate_layout:")
        traceback.print_exc()
        return None, {"error": str(e)}, f"Error: {str(e)}", None, None


# -----------------------------
# 🎨 UI
# -----------------------------
with gr.Blocks(title="SustainableUrbanPix2Pix v2.0") as demo:

    gr.Markdown(
        "# 🌆 SustainableUrbanPix2Pix v2.0\n"
        "Sketch → Sustainable Urban Layout + Analysis\n"
        "**100 Epochs • GTX 1050 Ti Optimized**"
    )

    with gr.Tabs():
        with gr.Tab("🎨 Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload High-Contrast Sketch (256×256)")
                    use_canny = gr.Checkbox(label="Apply Canny Edge Cleaning", value=True)
                    green_slider = gr.Slider(0, 1, value=0.6, label="Green Intensity")
                    density_slider = gr.Slider(0, 1, value=0.7, label="Building Density")

                    generate_btn = gr.Button("🚀 Generate Sustainable Layout", variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Sustainable Urban Layout")

        with gr.Tab("🎯 Metrics Dashboard"):
            with gr.Row():
                with gr.Column():
                    metrics_output = gr.JSON(label="Urban Metrics")
                with gr.Column():
                    insight_output = gr.Textbox(label="🧠 Planning Insights", lines=9)
            
            with gr.Row():
                radar_plot = gr.Plot(label="Sustainability Radar Chart")
            
            with gr.Row():
                download_json = gr.File(label="📥 Download Results as JSON")

    generate_btn.click(
        generate_layout,
        inputs=[input_image, green_slider, density_slider, use_canny],
        outputs=[output_image, metrics_output, insight_output, radar_plot, download_json]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)