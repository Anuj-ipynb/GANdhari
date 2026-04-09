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
# 🧠 Insight Engine (FIXED LOGIC)
# -----------------------------
def interpret(metrics):
    if not metrics or "error" in metrics:
        return "Error generating metrics."

    g = float(metrics.get("green_coverage", 0)) * 100
    d = float(metrics.get("building_density", 0)) * 100
    r = float(metrics.get("road_coverage", 0)) * 100
    score = float(metrics.get("sustainability_score", 50))
    conn = float(metrics.get("road_connectivity", 0))

    msg = []

    if g < 15:
        msg.append("⚠️ CRITICAL: Very low green coverage. Add parks, trees, and green corridors.")
    elif g < 30:
        msg.append("⚠️ WARNING: Moderate green space. Increase vegetation for better livability.")

    if d > 60:
        msg.append("⚠️ WARNING: High building density. Risk of overcrowding.")

    if r > 70:
        msg.append("⚠️ WARNING: Excessive road coverage. Reduce paved areas.")

    if conn < 5:
        msg.append("⚠️ WARNING: Poor road connectivity. Improve network continuity.")

    if score >= 75:
        msg.append("✅ EXCELLENT: Highly sustainable layout.")
    elif score >= 55:
        msg.append("✅ GOOD: Balanced layout with room for improvement.")
    else:
        msg.append("⚠️ NEEDS IMPROVEMENT: Increase green and reduce overbuilt areas.")

    return "\n".join(msg)


# -----------------------------
# 🔧 JSON SAFE CONVERSION
# -----------------------------
def make_json_safe(metrics):
    safe = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, (np.float32, np.float64)):
                safe[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                safe[k] = int(v)
            else:
                safe[k] = float(v)
        except Exception:
            safe[k] = 0.0
    return safe


# -----------------------------
# 🚀 MAIN PIPELINE (FIXED)
# -----------------------------
def generate_layout(sketch):
    if sketch is None:
        return None, {}, "", None, None

    try:
        # Save temp input
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp.name
        tmp.close()
        sketch.save(tmp_path)

        # 🔥 FIX: ONLY pass sketch path (NO canny / sliders)
        result_img, metrics = run_inference(tmp_path)

        if result_img is None or "error" in metrics:
            return None, metrics, "Inference failed.", None, None

        # JSON safe
        metrics = make_json_safe(metrics)

        insight = interpret(metrics)

        # -----------------------------
        # 📊 Radar Chart (FIXED SCALE)
        # -----------------------------
        fig = None
        try:
            cats = ["Green", "Road", "Density", "Connectivity"]

            vals = [
                metrics.get("green_coverage", 0) * 100,
                metrics.get("road_coverage", 0) * 100,
                100 - metrics.get("building_density", 0) * 100,
                min(metrics.get("road_connectivity", 0) * 2, 100),
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=420,
                title="Sustainability Radar"
            )
        except Exception as e:
            print("Radar error:", e)

        # -----------------------------
        # 💾 Save JSON
        # -----------------------------
        os.makedirs("outputs/results", exist_ok=True)
        json_path = "outputs/results/sustainability_results.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return result_img, metrics, insight, fig, json_path

    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()
        return None, {"error": str(e)}, str(e), None, None


# -----------------------------
# 🎨 UI (SIMPLIFIED + FIXED)
# -----------------------------
with gr.Blocks(title="Cityscapes Pix2Pix Generator") as demo:

    gr.Markdown(
        "# 🌆 Cityscapes Pix2Pix Generator\n"
        "Upload a **semantic map** → get a **realistic enhanced image + metrics**\n\n"
        "⚠️ Do NOT use edge images (Canny). Use segmentation maps only."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Semantic Map Input (256x256)")
            generate_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Output")

    with gr.Row():
        metrics_output = gr.JSON(label="Metrics")

    with gr.Row():
        insight_output = gr.Textbox(label="Insights", lines=8)

    with gr.Row():
        radar_plot = gr.Plot(label="Radar Chart")

    with gr.Row():
        download_json = gr.File(label="Download JSON")

    generate_btn.click(
        generate_layout,
        inputs=[input_image],
        outputs=[output_image, metrics_output, insight_output, radar_plot, download_json]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)