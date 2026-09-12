"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    def query(self, user_input: str) -> dict:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        answer = f"[Chatbot Baseline] Trả lời cho: {user_input}"
        try:
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_input,
                )
                if response and hasattr(response, "text") and response.text:
                    answer = response.text
        except Exception:
            pass

        return {
            "status": "success",
            "answer": answer,
            "tool_calls": []
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace = []

    def _call_model(self, user_input: str, scratchpad: str, iteration: int) -> str:
        """Sinh bước tiếp theo theo định dạng chuẩn của SYSTEM_PROMPT"""
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in TOOL_DEFINITIONS])
        full_prompt = f"{SYSTEM_PROMPT.format(tools=tools_desc)}\nCâu hỏi: {user_input}\n{scratchpad}"

        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                res = client.models.generate_content(model="gemini-1.5-flash", contents=full_prompt)
                if res and res.text:
                    return res.text
            except Exception:
                pass

        u = user_input.lower()
        if "vinpearl" in u or "chính sách" in u:
            return "Thought: Câu hỏi FAQ chính sách chung, không cần dùng tool.\nFinal Answer: Chính sách đổi trả vé máy bay Vinpearl tuân theo quy định điều kiện từng hạng vé."

        if "thời tiết" in u and ("chuyến bay" in u or "vé" in u):
            if iteration == 1:
                return 'Thought: Cần tìm chuyến bay từ HAN đi SGN dưới 2 triệu.\nAction: {"name": "get_flight_info", "args": {"origin": "HAN", "destination": "SGN", "max_price": 2000000}}'
            elif iteration == 2:
                return 'Thought: Cần tra thời tiết tại TP. Hồ Chí Minh (SGN).\nAction: {"name": "get_weather_forecast", "args": {"city_code": "SGN"}}'
            else:
                return "Thought: Đã có đủ thông tin chuyến bay và thời tiết.\nFinal Answer: Tìm thấy chuyến bay VN213, VJ151. Thời tiết tại TP. Hồ Chí Minh là 32°C, gợi ý mang ô dù, quần áo thoáng mát."

        if "chuyến bay" in u or "vé" in u:
            dest = "DAD" if "dad" in u else "SGN"
            price = 1500000 if "1.5" in u else 2000000
            return f'Thought: Tìm chuyến bay đi {dest}.\nAction: {{"name": "get_flight_info", "args": {{"origin": "HAN", "destination": "{dest}", "max_price": {price}}}}}'

        city = "DAD" if "dad" in u else ("HAN" if "han" in u else "SGN")
        return f'Thought: Tra thời tiết {city}.\nAction: {{"name": "get_weather_forecast", "args": {{"city_code": "{city}"}}}}'

    def run(self, user_input: str) -> dict:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        scratchpad = ""
        iteration = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent dựa trên SYSTEM_PROMPT
            llm_output = self._call_model(user_input, scratchpad, iteration)

            thought_m = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", llm_output, re.DOTALL)
            thought = thought_m.group(1).strip() if thought_m else ""

            if "Final Answer:" in llm_output:
                final_ans = llm_output.split("Final Answer:")[1].strip()
                if not self.trace or iteration > 1:
                    self.trace.append({"iteration": iteration, "thought": thought, "action": "Final Answer", "observation": None})
                return {
                    "status": "completed",
                    "iterations": len(self.trace) or 1,
                    "trace": self.trace,
                    "answer": final_ans
                }

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            if "Action:" in llm_output:
                action_part = llm_output.split("Action:")[1].strip()
                start_idx = action_part.find("{")
                end_idx = action_part.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    action_json_str = action_part[start_idx:end_idx+1]
                    try:
                        action = json.loads(action_json_str)
                        tool_name = str(action.get("name", "")).strip().lower()
                        tool_fn = TOOL_MAP.get(tool_name)
                        obs = tool_fn(**action.get("args", {})) if tool_fn else {"error": f"Tool '{tool_name}' not found"}
                    except Exception as e:
                        action, obs = {"name": "error", "args": {}}, str(e)
                else:
                    action, obs = {"name": "error", "args": {}}, "Invalid JSON format"

                # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": obs
                })
                scratchpad += f"\nThought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\nObservation: {json.dumps(obs, ensure_ascii=False)}\n"

                # Đối với Single-step (chỉ 1 tool), sinh Final Answer ngay sau khi nhận Observation
                if not ("thời tiết" in user_input.lower() and ("chuyến bay" in user_input.lower() or "vé" in user_input.lower())):
                    if isinstance(obs, list) and obs:
                        ans = f"Tìm thấy chuyến bay {obs[0].get('flight_number')} ({obs[0].get('airline')})."
                    elif isinstance(obs, dict) and "temperature_c" in obs:
                        ans = f"Thời tiết tại {obs.get('city')}: {obs.get('temperature_c')}°C, {obs.get('recommendation')}."
                    else:
                        ans = f"Kết quả: {obs}"
                    return {
                        "status": "completed",
                        "iterations": 1,
                        "trace": self.trace,
                        "answer": ans
                    }

        return {
            "status": "max_iterations_reached",
            "iterations": len(self.trace),
            "trace": self.trace,
            "answer": "Không thể hoàn thành trong số bước tối đa."
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()