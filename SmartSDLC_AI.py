import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024, temperature=0.7):
    """Generates a text response from the loaded model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def classify_requirements(raw_text):
    """
    Simulates the Requirement Upload and Classification feature.
    Takes raw text and uses the AI to classify sentences into SDLC phases.
    NOTE: This is a simulation. A real application would handle PDF files and
    more complex classification logic.
    """
    if not raw_text.strip():
        return "Please enter some text to classify."
        
    prompt = f"""
    The following text contains unstructured software requirements. Your task is to classify each sentence into one of the following SDLC phases: Requirements, Design, Development, Testing, or Deployment.

    After classifying, transform the output into a structured format, grouping the sentences by phase. If a sentence fits multiple phases, list it under the most relevant one.

    Raw Text:
    {raw_text}

    Structured Output:
    Requirements:
    - [Sentence 1]
    - [Sentence 2]
    ...
    Design:
    - [Sentence 3]
    ...
    Development:
    - [Sentence 4]
    ...
    Testing:
    - [Sentence 5]
    ...
    Deployment:
    - [Sentence 6]
    ...
    """

    response = generate_response(prompt, max_length=1500)
    return response

def generate_code(user_prompt):
    """
    Simulates the AI Code Generator feature.
    Takes a natural language prompt and generates a code snippet.
    """
    if not user_prompt.strip():
        return "Please enter a prompt to generate code."

    prompt = f"Generate a clean and well-commented code snippet in Python based on the following natural language prompt. Include a brief explanation of the code's functionality.\n\nPrompt: {user_prompt}\n\nCode:"

    response = generate_response(prompt, max_length=1000, temperature=0.5)
    
    # Extract just the code block if the model adds extra text
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # If no code block is found, just return the raw response
        code = response.strip()

    return code

def fix_bug(buggy_code):
    """
    Simulates the Bug Fixer feature.
    Takes a code snippet and returns a corrected, optimized version.
    """
    if not buggy_code.strip():
        return "Please paste some code to fix."
    
    prompt = f"""
    Analyze the following code snippet for syntax and logical errors. Provide a corrected, optimized, and well-commented version of the code.

    Original Code:
    {buggy_code}

    Corrected and Optimized Code:
    """
    
    response = generate_response(prompt, max_length=1000, temperature=0.3)
    
    # Extract just the code block
    code_match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # If no code block is found, just return the raw response
        code = response.strip()

    return code

# Create Gradio interface
with gr.Blocks(title="SmartSDLC") as app:
    gr.Markdown("# SmartSDLC: AI-Enhanced Software Development Lifecycle")
    gr.Markdown("SMARTSDLC is an AI-driven platform that automates SDLC tasks—like code generation, testing, and bug fixing—using advanced AI.")

    with gr.Tabs():
        with gr.TabItem("Requirement Classifier"):
            gr.Markdown("### Requirement Upload and Classification")
            gr.Markdown("Paste unstructured requirements below. The AI will classify them and organize them into user stories.")
            with gr.Row():
                with gr.Column():
                    raw_reqs = gr.Textbox(
                        label="Enter Raw Requirements (e.g., from a PDF)",
                        placeholder="The user shall be able to log in. The system must validate user credentials. The database schema needs to be designed. We will use a unit test framework to check all functions. The application will be deployed to a cloud server.",
                        lines=10
                    )
                    classify_btn = gr.Button("Classify & Transform")
                with gr.Column():
                    classified_output = gr.Textbox(label="Structured User Stories", lines=20)
            
            classify_btn.click(classify_requirements, inputs=raw_reqs, outputs=classified_output)

        with gr.TabItem("AI Code Generator"):
            gr.Markdown("### AI Code Generator")
            gr.Markdown("Describe the code you need in natural language, and the AI will generate it for you.")
            with gr.Row():
                with gr.Column():
                    code_prompt_input = gr.Textbox(
                        label="Describe the code to be generated",
                        placeholder="e.g., A Python function to calculate the factorial of a number using recursion.",
                        lines=5
                    )
                    generate_btn = gr.Button("Generate Code")
                with gr.Column():
                    code_output = gr.Code(label="Generated Code", language="python", lines=20)
            
            generate_btn.click(generate_code, inputs=code_prompt_input, outputs=code_output)

        with gr.TabItem("Bug Fixer"):
            gr.Markdown("### Bug Fixer")
            gr.Markdown("Paste a code snippet with a bug, and the AI will analyze and fix it.")
            with gr.Row():
                with gr.Column():
                    buggy_code_input = gr.Code(
                        label="Enter Buggy Code", 
                        language="python", 
                        lines=10,
                        value="""
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num < max_num:
            max_num = num
    return max_num
                        """
                    )
                    fix_btn = gr.Button("Fix Bug")
                with gr.Column():
                    fixed_code_output = gr.Code(label="Fixed Code", language="python", lines=10)
            
            fix_btn.click(fix_bug, inputs=buggy_code_input, outputs=fixed_code_output)

app.launch(share=True)
