import json
import os

files = [
    r"C:\Users\Aaron\.gemini\antigravity-ide\brain\266c7f89-0286-4987-abf9-ae5bc12c2a43\.system_generated\logs\transcript.jsonl",
    r"C:\Users\Aaron\.gemini\antigravity-ide\brain\dc31c4e7-c90d-4741-a5a4-e456c650f6e8\.system_generated\logs\transcript.jsonl"
]

out_dir = r"c:\Users\Aaron\Desktop\IIIT-Kottayam"

for fpath in files:
    if not os.path.exists(fpath): 
        print(f"File not found: {fpath}")
        continue
    
    chat_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fpath))))
    out_path = os.path.join(out_dir, f"recovered_chat_{chat_id}.md")
    
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"# Recovered Chat: {chat_id}\n\n")
        
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "USER_INPUT":
                        content = data.get("content", "")
                        if content:
                            out.write(f"### USER\n\n{content}\n\n---\n\n")
                    elif data.get("source") == "MODEL" and data.get("content"):
                        content = data.get("content", "")
                        # avoid dumping large tool calls or things that might not be text
                        if "tool_calls" not in data or data.get("type") == "PLANNER_RESPONSE" or data.get("type") == "STEP_RESPONSE":
                            out.write(f"### AI\n\n{content}\n\n---\n\n")
                except Exception as e:
                    pass
    print(f"Created {out_path}")
