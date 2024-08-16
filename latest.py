#CONTAINS ALL FUNCTIONALITIES

from flask import Flask, render_template, request, jsonify
from neo4j import GraphDatabase
from pyvis.network import Network
import random
import ssl
import json
from datetime import datetime
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
from flask_cors import CORS
import textwrap
from langchain_openai import ChatOpenAI
import os


app = Flask(__name__)
CORS(app)


# Neo4j connection setup
uri = "neo4j://6cb5ec24.databases.neo4j.io"
user = "neo4j"
password = "Ww3XWwg1qPUkKzeCoxmBGeM0Jw9pY3HZLQ2V7vzQwjA"

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    encrypted=True,
    trust="TRUST_ALL_CERTIFICATES",
    ssl_context=ssl_context,
    connection_timeout=30,
    max_connection_lifetime=3600
)

# Define colors for each type
label_colors = {
    'ANALYSIS': '#1f77b4',        # Steel Blue
    'ARGUMENTS': '#ff7f0e',       # Dark Orange
    'CASE_NO': '#2ca02c',         # Forest Green
    'COURT': '#d62728',           # Brick Red
    'DATE': '#9467bd',            # Medium Purple
    'DECISION': '#8c564b',        # Brown
    'DECISION_OF_JUDGEMENT': '#e377c2',  # Orchid
    'FACTS': '#FFFF00',           # yellow
    'GPE': '#bcbd22',             # Olive
    'GROUNDS': '#17becf',         # Deep Sky Blue
    'JUDGE': '#ff9896',           # Light Salmon
    'LAWYER': '#98df8a',          # Pale Green
    'ORDER': '#c5b0d5',           # Thistle
    'PARTICULAR': '#ffbb78',      # Light Orange
    'PETITIONER': '#dbdb8d',      # Khaki
    'PRAYER': '#c49c94',          # Rosy Brown
    'PRE_RELIED': '#9edae5',      # Powder Blue
    'PRECEDENT': '#f7b6d2',       # Pink
    'PROVISION': '#c7c7c7',       # Silver
    'RESPONDENT': '#aec7e8',      # Light Steel Blue
    'RLC': '#ff9f55',             # Sandy Brown
    'STATUTE': '#66c2a5',         # Medium Aquamarine
    'SUBJECT_MATTER': '#e6ab02',  # Goldenrod
    'WITNESS': '#8da0cb'          # Cornflower Blue
}


def create_graph(query):
    net = Network(height='800px', width='100%', bgcolor='#FAF9F6', font_color='black')

    physics_options = {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "theta": 0.5,
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 600,
                "springConstant": 0.1,
                "damping": 0.09,
                "avoidOverlap": 1
            },
        }
    }

    net.set_options(json.dumps(physics_options))

    with driver.session() as session:
        result = session.run(query)
        records = list(result)  # Consume all records within the session context

    for record in records:
        n = record["n"]
        r = record["r"]
        m = record["m"]

        n_name = str(n.get("name", ""))
        m_name = str(m.get("name", ""))
        n_labels = list(n.labels)
        m_labels = list(m.labels)
        r_type = type(r).__name__

        def truncate_label(label):
            return label if len(label) <= 20 else label[:20] + "..."

        def wrap_text(text, width):
            return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))

        n_name_display = truncate_label(n_name)
        m_name_display = truncate_label(m_name)

        n_color = label_colors.get(n_labels[0], 'lightgray') if n_labels else 'lightgray'
        m_color = label_colors.get(m_labels[0], 'lightgray') if m_labels else 'lightgray'

        net.add_node(n_name, label=n_name_display, title="Type: "+ str(n_labels)+"\n Description: "+ wrap_text(n_name, 40), color=n_color)
        net.add_node(m_name, label=m_name_display, title="Type: "+ str(m_labels)+"\n Description: "+ wrap_text(m_name, 40), color=m_color)
        net.add_edge(n_name, m_name, title=r_type, color='#A9A9A9')

    timestamp = datetime.now().strftime("%H%M%S")
    output_path = f'saved_graphs/knowledge_graph_{timestamp}.html'
    net.save_graph(output_path)

    return net


@app.route('/')
def index():
    queries = {
        'case_nos': "MATCH (n:CASE_NO) RETURN DISTINCT n.name AS name",
        'provisions': "MATCH (n:PROVISION) RETURN DISTINCT n.name AS name",
        'subject_matters': "MATCH (n:SUBJECT_MATTER) RETURN DISTINCT n.name AS name",
        'judges': "MATCH (n:JUDGE) RETURN DISTINCT n.name AS name"
    }
    
    results = {}
    with driver.session() as session:
        for key, query in queries.items():
            result = session.run(query)
            results[key] = [record["name"] for record in result]

    return render_template('final_index.html', **results)

@app.route('/overall_graph', methods=['GET'])
def overall_graph():
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT 1000
    """
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/search_graph', methods=['POST'])
def search_graph():
    case_nos = request.form.getlist('case_nos')
    provisions = request.form.getlist('provisions')
    subject_matters = request.form.getlist('subject_matters')
    judges = request.form.getlist('judges')

    conditions = []
    if case_nos:
        conditions.append(f"n:CASE_NO AND n.name IN {case_nos}")
    if provisions:
        conditions.append(f"n:PROVISION AND n.name IN {provisions}")
    if subject_matters:
        conditions.append(f"n:SUBJECT_MATTER AND n.name IN {subject_matters}")
    if judges:
        conditions.append(f"n:JUDGE AND n.name IN {judges}")

    where_clause = " OR ".join(conditions)
    
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE {where_clause}
    RETURN n, r, m
    LIMIT 1000
    """
    
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/search_text', methods=['POST'])
def search_text():
    text = request.form.get('text')
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE n.name CONTAINS '{text}' OR m.name CONTAINS '{text}' 
    RETURN n, r, m
    LIMIT 1000
    """
    graph_html = create_graph(query)
    return graph_html.generate_html()

@app.route('/add_node', methods=['POST'])
def add_node():
    node1_type = request.form.get('node1_type')
    node1_name = request.form.get('node1_name')
    relation_name = request.form.get('relation_name')
    node2_type = request.form.get('node2_type')
    node2_name = request.form.get('node2_name')

    query = f"""
    MERGE (n1:{node1_type} {{name: '{node1_name}'}})
    MERGE (n2:{node2_type} {{name: '{node2_name}'}})
    MERGE (n1)-[r:{relation_name}]->(n2)
    """
    with driver.session() as session:
        session.run(query)
    
    return jsonify({'status': 'Node and relationship added'})

@app.route('/delete_node', methods=['POST'])
def delete_node():
    node_type = request.form.get('node_type')
    node_name = request.form.get('node_name')

    query = f"""
    MATCH (n:{node_type} {{name: '{node_name}'}})
    DETACH DELETE n
    """
    with driver.session() as session:
        session.run(query)
    
    return jsonify({'status': 'Node deleted'})

@app.route('/graph_qa', methods=['POST'])
def graph_qa():
    query = request.form.get('query')
    # groq_api_key="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"
    os.environ["OPENAI_API_KEY"] = "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"
    url = "neo4j+s://6cb5ec24.databases.neo4j.io"
   
    graph=Neo4jGraph(
        url=url,
        username=user,
        password=password
    )
    # llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    llm = ChatOpenAI(model_name="gpt-4o")

    chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

    response = chain.invoke({"query": query})
    response = response.get('result', 'No result found')


    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
