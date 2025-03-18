# %%
from io import BytesIO
from time import sleep
import os
import pathlib
from huggingface_hub import login

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import (
    CodeAgent, 
    tool, 
    HfApiModel, 
    DuckDuckGoSearchTool,
    LiteLLMModel
)
from smolagents.agents import ActionStep

# 환경 변수 로드 및 설정
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

# 글로벌 드라이버 변수
driver = None

@tool
def visit_webpage(url: str) -> str:
    """Visit a webpage at the specified URL.
    
    Args:
        url: The URL to visit
    
    Returns:
        A confirmation message
    """
    helium.go_to(url)
    sleep(0.5)  # 페이지 로딩을 위한 대기 시간
    return f"Visited {url}"

@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    global driver
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if not elements:
        return f"No matches found for '{text}'"
    if nth_result > len(elements):
        return f"Only found {len(elements)} matches for '{text}', cannot jump to match #{nth_result}"
    
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    driver.execute_script("arguments[0].style.border='3px solid red'", elem)  # 강조 표시
    sleep(0.5)
    return result + f" Focused on element {nth_result} of {len(elements)}"

@tool
def go_back() -> str:
    """Goes back to previous page."""
    driver.back()
    sleep(1.5)  # 페이지 로딩을 위한 대기 시간
    return "Navigated back to previous page"

@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    sleep(0.5)
    return "Attempted to close any open popups"

@tool
def click_element(element_text: str) -> str:
    """
    Clicks on an element with the specified text.
    
    Args:
        element_text: The text of the element to click
        
    Returns:
        A confirmation message
    """
    try:
        helium.click(element_text)
        sleep(1)  # 클릭 후 로딩 대기
        return f"Clicked on element with text: '{element_text}'"
    except Exception as e:
        return f"Failed to click on '{element_text}': {str(e)}"

@tool
def click_link(link_text: str) -> str:
    """
    Clicks on a link with the specified text.
    
    Args:
        link_text: The text of the link to click
        
    Returns:
        A confirmation message
    """
    try:
        helium.click(helium.Link(link_text))
        sleep(1)  # 클릭 후 로딩 대기
        return f"Clicked on link: '{link_text}'"
    except Exception as e:
        return f"Failed to click on link '{link_text}': {str(e)}"

@tool
def scroll_down(pixels: int = 500) -> str:
    """
    Scrolls down the page by the specified number of pixels.
    
    Args:
        pixels: Number of pixels to scroll down
        
    Returns:
        A confirmation message
    """
    helium.scroll_down(num_pixels=pixels)
    sleep(0.5)
    return f"Scrolled down {pixels} pixels"

@tool
def scroll_up(pixels: int = 500) -> str:
    """
    Scrolls up the page by the specified number of pixels.
    
    Args:
        pixels: Number of pixels to scroll up
        
    Returns:
        A confirmation message
    """
    helium.scroll_up(num_pixels=pixels)
    sleep(0.5)
    return f"Scrolled up {pixels} pixels"

@tool
def type_text(text: str) -> str:
    """
    Types the specified text into the active/focused element.
    
    Args:
        text: The text to type
        
    Returns:
        A confirmation message
    """
    helium.write(text)
    return f"Typed: '{text}'"

@tool
def press_key(key: str) -> str:
    """
    Presses a specific key (like Enter, Tab, etc).
    
    Args:
        key: The key to press ('ENTER', 'TAB', 'SPACE', 'ARROW_DOWN', 'ARROW_UP', etc)
        
    Returns:
        A confirmation message
    """
    key_map = {
        "ENTER": Keys.RETURN,
        "TAB": Keys.TAB,
        "SPACE": Keys.SPACE,
        "ARROW_DOWN": Keys.ARROW_DOWN,
        "ARROW_UP": Keys.ARROW_UP,
        "ARROW_LEFT": Keys.ARROW_LEFT,
        "ARROW_RIGHT": Keys.ARROW_RIGHT,
        "ESCAPE": Keys.ESCAPE,
    }
    
    if key.upper() in key_map:
        webdriver.ActionChains(driver).send_keys(key_map[key.upper()]).perform()
        return f"Pressed key: {key}"
    else:
        return f"Unknown key: {key}. Please use one of {list(key_map.keys())}"

@tool
def search_web(query: str) -> str:
    """
    Searches for information using DuckDuckGo (no CAPTCHA issues).
    
    Args:
        query: The search query
        
    Returns:
        Search results
    """
    try:
        # URL 패턴인지 확인
        import re
        url_pattern = re.compile(r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        match = url_pattern.match(query)
        if match:
            domain = match.group(3)
            if not query.startswith(('http://', 'https://')):
                return visit_webpage(f"https://{query if query.startswith('www.') else 'www.' + query}")
            else:
                return visit_webpage(query)
        
        # 단순 도메인 형태인지 확인
        domain_pattern = re.compile(r'^([a-zA-Z0-9-]+)$')
        if domain_pattern.match(query):
            return visit_webpage(f"https://www.{query}.com")
        
        # DuckDuckGo 검색 API 사용
        search_tool = DuckDuckGoSearchTool()
        results = search_tool(query)
        
        # 결과가 있으면 첫 번째 URL 방문 시도
        top_result_url = None
        
        # 결과에서 URL 추출 시도
        try:
            import json
            if isinstance(results, str):
                # JSON 형태로 파싱 시도
                try:
                    parsed_results = json.loads(results)
                    if isinstance(parsed_results, list) and len(parsed_results) > 0:
                        if isinstance(parsed_results[0], dict) and 'url' in parsed_results[0]:
                            top_result_url = parsed_results[0]['url']
                except:
                    # 일반 텍스트에서 URL 추출 시도
                    url_matches = re.findall(r'https?://[^\s"\'<>]+', results)
                    if url_matches:
                        top_result_url = url_matches[0]
        except:
            pass
            
        if top_result_url:
            visit_webpage(top_result_url)
            return f"Found result for '{query}' and navigated to top result: {top_result_url}"
        
        # DuckDuckGo 직접 방문을 통한 검색
        visit_webpage("https://duckduckgo.com/")
        sleep(1)
        
        try:
            # 검색창 찾기 시도
            global driver
            search_box = driver.find_element(By.ID, "searchbox_input")
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            sleep(1)
            return f"Searched DuckDuckGo for: '{query}'"
        except Exception as e:
            try:
                # 대체 방법: name 속성으로 검색
                search_box = driver.find_element(By.NAME, "q")
                search_box.clear()
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                sleep(1)
                return f"Searched DuckDuckGo for: '{query}'"
            except Exception as e2:
                return f"DuckDuckGo search results: {results}"
                
    except Exception as e:
        return f"Failed to search or navigate: {str(e)}. Try with a specific URL using visit_webpage."

def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # 자바스크립트 애니메이션 완료 대기
    global driver
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # 이전 스크린샷 제거
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # 복사본 생성

    # 현재 URL 정보 업데이트
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )

def main():
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
    
    # 사용자 입력 받기
    user_input = input("Enter your prompt: ")
    
    # 모델 생성
    if model_choice == "2":
        model = HfApiModel(model_id=model_id)
    else:
        model = LiteLLMModel(model_id=model_id, api_key=api_key)

    # Chrome 옵션 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1.5")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")

    # 브라우저 초기화
    global driver
    driver = helium.start_chrome(headless=False, options=chrome_options)

    # 에이전트 생성
    agent = CodeAgent(
        tools=[
            visit_webpage,
            search_item_ctrl_f,
            go_back,
            close_popups,
            click_element,
            click_link,
            scroll_down,
            scroll_up,
            type_text,
            press_key,
            search_web
        ],
        model=model,
        additional_authorized_imports=["time", "re"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )

    # 브라우징 지시사항
    browsing_instructions = """
You are a web browsing assistant that can see and interact with webpages. You have access to a Chrome browser and can see screenshots of the current page.

Your task is to help the user by navigating webpages and finding information they need. Here are the tools you can use:

1. DIRECT NAVIGATION - USE YOUR KNOWLEDGE:
   - When you know the exact URL, use: visit_webpage("https://www.example.com")
   - You have knowledge of many common website domains - use this knowledge!
   - Examples:
     * If user asks about Python, you know it's at: visit_webpage("https://www.python.org")
     * If user wants documentation, you know common URLs like: visit_webpage("https://docs.python.org")
     * For popular sites like Reddit, GitHub, Twitter, etc., use your own knowledge of their URLs
   - Only use search as a fallback if you genuinely don't know the URL

2. INTERACTION:
   - click_element("text") - Click on an element with this text
   - click_link("text") - Click specifically on a link
   - type_text("text") - Type text into a field
   - press_key("key") - Press a key (ENTER, TAB, etc.)
   - close_popups() - Close popup windows

3. NAVIGATION:
   - scroll_down(pixels) - Scroll down (default: 500px)
   - scroll_up(pixels) - Scroll up (default: 500px)
   - search_item_ctrl_f("text") - Find text on the page
   - go_back() - Go back to the previous page

4. SEARCH (only as fallback):
   - search_web("query") - Use DuckDuckGo to search (no CAPTCHA issues)
   - You can also try search_web with a domain name if you're unsure of the exact URL format

IMPORTANT TIPS:
- USE YOUR KNOWLEDGE FIRST - You already know many website URLs, use them directly
- Look at the screenshot to understand the page content
- Report what you see and what you're planning to do
- Be patient with page loading
- If you encounter a CAPTCHA, try another approach or site

Now, I'll help you with: 
"""

    try:
        # 에이전트 실행
        agent_output = agent.run(browsing_instructions + user_input)
        print("\nFinal output:", agent_output)
    except Exception as e:
        print(f"\nError during agent execution: {str(e)}")
    finally:
        print("\nFinished...")

if __name__ == "__main__":
    main()
# %%
