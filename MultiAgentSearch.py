# %%
from huggingface_hub import login
from dotenv import load_dotenv
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
import os
from langdetect import detect
import pathlib
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    MultiStepAgent
)

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / '.env')

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model_id = "gemini/gemini-2.0-flash-lite"

hf_model_id = "Qwen/Qwen2.5-72b-Instruct"
hf_token = os.environ.get("HF_TOKEN")

claude_api_key = os.environ.get("CLAUDE_API_KEY")
claude_model_id = "claude-3-7-sonnet-20250219" # "claude-3-5-sonnet-20241022"

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_model_id = "gpt-4o-mini"

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def translate_to_english(user_prompt: str) -> str:
    """Translates the given text from the detected language to English.
    
    Args:
        user_prompt: The text to be translated to English.
        
    Returns:
        The translated text in English.
    """
    global model_choice, model_id, api_key
    
    if model_choice == "2":
        temp_model = HfApiModel(model_id=model_id)
    else:
        temp_model = LiteLLMModel(model_id=model_id, api_key=api_key)

    messages = [
        {"role": "user", "content": f"Please accurately translate this text to English, preserving all meaning: '{user_prompt}'"}
    ]
    response = temp_model(messages)
    return response.content

@tool
def translate_to_target_language(english_text: str, target_language: str) -> str:
    """Translates the given English text to the specified target language.
    
    Args:
        english_text: The English text to be translated.
        target_language: The target language code (e.g., 'ko', 'ja', 'zh-cn').
        
    Returns:
        The translated text in the target language.
    """
    global model_choice, model_id, api_key
    
    if model_choice == "2":
        temp_model = HfApiModel(model_id=model_id)
    else:
        temp_model = LiteLLMModel(model_id=model_id, api_key=api_key)

    messages = [
        {"role": "user", "content": f"Please accurately translate this English text to {target_language}, preserving all meaning: '{english_text}'"}
    ]
    response = temp_model(messages)
    return response.content

def main():
    global model_choice, model_id, api_key
    
    print("Model Selection:")
    print("1. Gemini-2.0-flash-lite")
    print("2. Hugging Face (Qwen 2.5-72b-Instruct)")
    print("3. Claude-3-7-sonnet-20250219")
    print("4. OpenAI (gpt-4o-mini)")
    
    model_choice = input("Enter your choice (1-4): ")

    if model_choice == "1":
        model_id = gemini_model_id
        api_key = gemini_api_key
    elif model_choice == "2":
        model_id = hf_model_id
        api_key = hf_token
    elif model_choice == "3":
        model_id = claude_model_id
        api_key = claude_api_key
    elif model_choice == "4":
        model_id = openai_model_id
        api_key = openai_api_key
    else:
        print("Invalid choice. Defaulting to Gemini.")
        model_id = gemini_model_id
        api_key = gemini_api_key
    
    user_input = input("Enter your query: ")
    
    # Detect the language of the user input
    try:
        detected_input_language = detect(user_input)
        print(f"Detected language: {detected_input_language}")
    except:
        detected_input_language = "en"
        print("Could not detect language, defaulting to English")
    
    # Create the model for our agents
    if model_choice == "2":
        model = HfApiModel(model_id=model_id)
    else:
        model = LiteLLMModel(model_id=model_id, api_key=api_key)
    
    # Create the web search agent
    web_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), visit_webpage],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Runs web searches and visits webpages to find information.",
    )
    
    # Create the translation agent
    translate_agent = ToolCallingAgent(
        tools=[translate_to_english, translate_to_target_language],
        model=model,
        max_steps=3,  # Reduced max steps to prevent excessive iterations
        name="translate_agent",
        description=f"Translates content between languages. Used for translating user queries to English and translating responses back to the user's language.",
    )
    
    # Create the manager agent with a more explicit system prompt
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent, translate_agent],
        additional_authorized_imports=["time", "numpy", "pandas"],
    )
    answer = manager_agent.run(user_input)
    #print("Assistant:", answer)
    
    # Detect the language of the answer
    try:
        detected_output_language = detect(answer)
        print(f"Detected language: {detected_output_language}")
    except:
        detected_output_language = "en"
        print("Could not detect language, defaulting to English")
    
    # Create a more explicit system prompt with clear step-by-step instructions
    if detected_input_language != detected_output_language:
        # For non-English queries, use a translation workflow
        translated_answer = translate_agent.run(f"Translate this text accurately to {detected_input_language}: '{answer}'")
        print(f"Translated answer: {translated_answer}")
    else:
        # For English queries, skip translation
        print("Assistant:", answer)


if __name__ == "__main__":
    main()

# %%
