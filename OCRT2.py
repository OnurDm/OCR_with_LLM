from paddleocr import PaddleOCR
from PIL import Image
import gradio as gr
import os
import shutil
from docx import Document
import uuid
import json
import zipfile
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')



def ocr_recognition_bulk(images, output_format):
    """
    Perform OCR on the input images in bulk, correct the text using OpenAI's GPT-4,
    return the corrected texts, and provide a downloadable zip file.
    """
    # Create a unique directory for the batch processing
    batch_id = str(uuid.uuid4())
    os.makedirs(batch_id, exist_ok=True)

    file_paths = []
    corrected_texts = []
    for idx, image in enumerate(images):
        # Perform OCR
        result = ocr_model.ocr(image)

        # Extract text from the result
        formatted_text = []
        for line in result:
            for word_info in line:
                formatted_text.append(word_info[1][0])

        # Combine the text into a single string
        formatted_text_output = '\n'.join(formatted_text)

        # Prepare the message for OpenAI
        message = """
Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:
1. Fix OCR-induced typos and errors:
- Correct words split across line breaks
- Fix common OCR errors (e.g., 'rn' misread as 'm', 'e' as 'c', 'h' as 'li', and 'f' as 'r')
- Use context and common sense to correct errors
- Only fix clear errors, don't alter the content unnecessarily
- Do not add extra periods or any unnecessary punctuation
2. Maintain original structure:
- Keep all headings and subheadings intact
3. Maintain coherence:
- Ensure the content connects smoothly with the previous context
- Handle text that starts or ends mid-sentence appropriately
After you correct the mistakes, format the text in markdown.
When appropriate, place content in tables (e.g., when a word is followed by x, consider this a check and place this x and the value both in a table).
Don't add any new text at the beginning or at the end, such as 'Document 1:'.
Text to correct:
"""
        message += formatted_text_output

        # Generate corrected text using OpenAI's GPT-4
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=0.5,
        )

        corrected_text = response.choices[0].message.content

        # Append the corrected text to the list
        corrected_texts.append(f"Document {idx + 1}:\n{corrected_text}\n\n")

        # Generate a unique filename for each file
        unique_id = str(uuid.uuid4())
        if output_format == '.txt':
            output_file = os.path.join(batch_id, f'corrected_text_{unique_id}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
        elif output_format == '.docx':
            output_file = os.path.join(batch_id, f'corrected_text_{unique_id}.docx')
            doc = Document()
            doc.add_paragraph(corrected_text)
            doc.save(output_file)
        elif output_format == '.json':
            output_file = os.path.join(batch_id, f'corrected_text_{unique_id}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'corrected_text': corrected_text}, f, ensure_ascii=False, indent=4)
        else:
            return "Invalid output format selected.", None

        file_paths.append(output_file)

    # Create a zip file containing all the output files
    zip_file_path = f'{batch_id}/corrected_texts_{batch_id}.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))

    # Combine all corrected texts into a single string for display
    display_text = ''.join(corrected_texts)

    # Return the combined corrected text and the path to the zip file
    return display_text, zip_file_path

# Set up the Gradio frontend
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Bulk Text Extractor</h1>")
    gr.Markdown("""
    <p style='text-align: center;'>
    Upload multiple images to extract text using PaddleOCR, correct it using OpenAI's GPT-4, and get the corrected text in your preferred format.
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(type="filepath", file_count="multiple", label="Upload Images (multiple files supported)")
            format_selection = gr.Radio(['.txt', '.docx', '.json'], label='Output Format', value='.txt')
            submit_button = gr.Button("Extract Text")
        with gr.Column(scale=1):
            corrected_text_output = gr.Textbox(label="Corrected Texts", lines=20)
            download_file = gr.File(label="Download Processed Files")

    submit_button.click(fn=ocr_recognition_bulk, inputs=[image_input, format_selection], outputs=[corrected_text_output, download_file])

# Launch 
demo.launch()
