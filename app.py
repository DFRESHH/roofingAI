from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class MAI:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.assistant_id = os.getenv('ASSISTANT_ID')
        self.threads = {}
    
    def process_message(self, user_id: str, message: str) -> str:
        try:
            thread = self._get_or_create_thread(user_id)
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            is_first_message = len(messages.data) == 0
            
            if is_first_message:
                initial_question = "Fantastic, glad you are here! What would you like to know about MAI?"
                self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="assistant",
                    content=initial_question
                )
                return initial_question
            
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message
            )
            
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            response = self._wait_for_response(thread.id, run.id, timeout=25)  # Cap at 25s
            return response
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return "Sorry, something went wrong."
    
    def _get_or_create_thread(self, user_id: str):
        if user_id not in self.threads:
            thread = self.client.beta.threads.create()
            self.threads[user_id] = thread
        return self.threads[user_id]
    
    def _wait_for_response(self, thread_id: str, run_id: str, timeout: int) -> str:
        start_time = time.time()
        while time.time() - start_time < timeout:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                return messages.data[0].content[0].text.value
            elif run.status in ["failed", "cancelled"]:
                return "Sorry, I couldn't process that."
            time.sleep(0.5)  # Poll faster but lighter
        return "Sorry, taking too long—try again later."

mai = MAI()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('user_id', 'default_user')
    message = data.get('message', '')
    response = mai.process_message(user_id, message)
    return jsonify({'response': response})

@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json()
    user_id = data.get('user_id', 'default_user')
    
    # Clear the thread from memory
    if user_id in mai.threads:
        del mai.threads[user_id]
    
    return jsonify({'status': 'reset successful'})

if __name__ == '__main__':
    # Print startup message
    print("Starting MAI Training Interface...")
    print(f"OpenAI API Key: {'Present' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
    print(f"Assistant ID: {'Present' if os.getenv('ASSISTANT_ID') else 'Missing'}")
    
    # Run the app
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)  # Render uses PORT
    