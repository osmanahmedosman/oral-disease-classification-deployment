import os

# Use a safe temporary folder for Gradio uploads
os.environ["GRADIO_TEMP_DIR"] = r"C:\gradio_temp"
os.makedirs(r"C:\gradio_temp", exist_ok=True)

import html
import numpy as np
import tensorflow as tf
import gradio as gr


MODEL_PATH = "models/oral_disease_classifier.keras"

CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration",
]


def preprocess_image(image):
    """Prepare an uploaded image for the trained model."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image, axis=0)


def empty_result():
    return """
    <div class="empty-result">
        Upload an image and click Analyze Image
        to view the classification results.
    </div>
    """


def predict(image):
    if image is None:
        return empty_result()

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)[0]

    if len(predictions) != len(CLASS_NAMES):
        return (
            "<div class='empty-result'>"
            "Error: Model output does not match the class list."
            "</div>"
        )

    results = sorted(
        zip(CLASS_NAMES, predictions),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    top_class, top_probability = results[0]
    top_percentage = float(top_probability) * 100

    result_html = f"""
    <div class="results-container">
        <h2>Classification Results</h2>
        <p class="results-subtitle">Top Prediction</p>

        <div class="top-prediction">
            <div class="top-class">
                {html.escape(top_class)}
            </div>
            <div class="top-confidence">
                {top_percentage:.2f}%
            </div>
            <div class="top-caption">
                Highest model prediction score
            </div>
        </div>

        <h3 class="all-results-title">All Class Scores</h3>
    """

    for index, (class_name, probability) in enumerate(results):
        percentage = float(probability) * 100

        if index == 0:
            color = "#16A34A"
            status = "Top prediction"
        elif percentage >= 10:
            color = "#D97706"
            status = "Other possible class"
        else:
            color = "#965C1A"
            status = "Lower score"

        result_html += f"""
        <div class="result-item">
            <div class="result-heading">
                <span class="class-name">
                    <span
                        class="status-dot"
                        style="background-color: {color};">
                    </span>
                    {html.escape(class_name)}
                </span>

                <span
                    class="confidence"
                    style="color: {color};">
                    {percentage:.2f}%
                </span>
            </div>

            <div class="progress-track">
                <div
                    class="progress-fill"
                    style="width: {max(0, min(100, percentage)):.2f}%;
                           background-color: {color};">
                </div>
            </div>

            <div class="result-status">{status}</div>
        </div>
        """

    result_html += """
        <div class="medical-note">
            <strong>Important notice</strong><br>
            This application is intended for educational and research
            purposes only. Model predictions are not a medical diagnosis.
            Consult a qualified dental professional for medical concerns.
        </div>
    </div>
    """

    return result_html


# Load the trained model once when the app starts
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        "The number of model outputs does not match CLASS_NAMES."
    )

print("Model loaded successfully!")


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.main-title {
    text-align: center;
    margin-bottom: 5px;
}

.main-subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
}

.results-container {
    padding: 22px;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: var(--background-fill-primary);
}

.results-container h2 {
    margin-top: 0;
    margin-bottom: 6px;
}

.results-subtitle {
    color: #64748b;
    font-size: 14px;
    margin: 0 0 12px;
}

.top-prediction {
    padding: 18px;
    margin-bottom: 24px;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    background: #f0fdf4;
    text-align: center;
}

.top-class {
    color: #166534;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
}

.top-confidence {
    color: #16a34a;
    font-size: 32px;
    font-weight: 800;
}

.top-caption {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.all-results-title {
    font-size: 16px;
    margin: 0 0 18px;
}

.result-item {
    margin-bottom: 22px;
}

.result-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.class-name {
    display: flex;
    align-items: center;
    gap: 9px;
    font-weight: 600;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.confidence {
    font-weight: 700;
    font-size: 16px;
}

.progress-track {
    width: 100%;
    height: 9px;
    background: #e2e8f0;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
}

.result-status {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.medical-note {
    border-top: 1px solid #e2e8f0;
    padding-top: 16px;
    margin-top: 24px;
    color: #64748b;
    font-size: 12px;
    line-height: 1.7;
}

.empty-result {
    padding: 28px 18px;
    text-align: center;
    color: #64748b;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css=custom_css,
    title="Oral Disease Classifier",
) as demo:

    gr.Markdown(
        """
        <div class="main-title">
            <h1>Oral Disease Classifier</h1>
        </div>
        <div class="main-subtitle">
            Deep Learning-Based Oral Disease Classification
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Oral Image",
                height=350,
            )

            with gr.Row():
                predict_button = gr.Button(
                    "Analyze Image",
                    variant="primary",
                )

                clear_button = gr.ClearButton(
                    components=[image_input],
                    value="Clear",
                )

        with gr.Column(scale=1):
            prediction_output = gr.HTML(
                value=empty_result(),
                label="Classification Results",
            )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=prediction_output,
    )

    clear_button.click(
        fn=empty_result,
        inputs=None,
        outputs=prediction_output,
    )


if __name__ == "__main__":
    demo.launch()