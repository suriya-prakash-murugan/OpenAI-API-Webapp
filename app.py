from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_url_path='', static_folder='static')

# Load the CSV data
df = pd.read_csv('static/Test data.csv')

# Initialize the model
model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.6)

# Create a global agent
global agent
agent = create_pandas_dataframe_agent(
    model,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

@app.route('/')
def serve_index():
    table_html = df.to_html(classes='data table table-bordered table-hover', index=False)
    return render_template('index.html', table_html=table_html)

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory(app.static_folder, path)

@app.route('/summary', methods=['POST'])
def summary():
    summary_text = agent.run("Analyze and provide the entire summary of the data")
    print(summary_text)
    return jsonify(summary_text=summary_text)

if __name__ == '__main__':
    print("Serving Initializing")
    with app.app_context():
        print("Serving Started")
        app.run(host="0.0.0.0", debug=True, port=9001)
