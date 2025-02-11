from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import ast
import torch
import time
import logging
import numpy as np
import shutil
from git import Repo
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(title="Code Analysis API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = "neo4j+s://da4f9389.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "DzwOi_LChEKK7mniHJDq5f-1a3GW3KSs7r8-vZxuzfw"

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize ML models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
    api_key=GROQ_API_KEY
)

# Create conversation memory
conversation_memory = ConversationBufferMemory(return_messages=True)

# Pydantic models
class RepoRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    query: str

class FunctionResponse(BaseModel):
    name: str
    description: Optional[str]
    body: str
    similarity_score: float
    file_path: Optional[str]

class ChatResponse(BaseModel):
    response: str
    context: Optional[Dict[str, Any]] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str

# Helper functions
def clear_directory(directory: str):
    """Clear a directory if it exists."""
    if os.path.exists(directory):
        shutil.rmtree(directory)

def clear_neo4j_database():
    """Deletes all nodes and relationships in the Neo4j database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Successfully cleared Neo4j database")

def clone_repo(repo_url: str) -> str:
    """Clone a repository and return the target directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target_dir = f"cloned_repo_{timestamp}"
    
    try:
        Repo.clone_from(repo_url, target_dir)
        logger.info(f"Successfully cloned repository to {target_dir}")
        return target_dir
    except Exception as e:
        logger.error(f"Failed to clone repository: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to clone repository")

def extract_functions(file_path: str) -> List[Dict[str, Any]]:
    """Extract function definitions from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node) or "No docstring",
                    "body": ast.get_source_segment(content, node)
                })
        return functions
    except Exception as e:
        logger.error(f"Failed to extract functions from {file_path}: {str(e)}")
        return []

def process_directory(directory: str, driver: GraphDatabase.driver):
    """Process a directory and store its structure in Neo4j."""
    try:
        with driver.session() as session:
            for root, dirs, files in os.walk(directory):
                # Filter out unwanted directories and files
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                files = [f for f in files if not f.startswith('.')]
                
                root_path = os.path.abspath(root)
                
                # Create directory node
                session.run(
                    "MERGE (d:Directory {path: $path}) SET d.name = $name",
                    path=root_path,
                    name=os.path.basename(root)
                )
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Create file node
                    session.run(
                        """
                        MERGE (f:File {path: $path})
                        SET f.name = $name,
                            f.size = $size
                        """,
                        path=file_path,
                        name=file,
                        size=os.path.getsize(file_path)
                    )
                    
                    # Create relationship between directory and file
                    session.run(
                        """
                        MATCH (d:Directory {path: $dir_path})
                        MATCH (f:File {path: $file_path})
                        MERGE (d)-[:CONTAINS]->(f)
                        """,
                        dir_path=root_path,
                        file_path=file_path
                    )
                    
                    # Process Python files
                    if file.endswith('.py'):
                        functions = extract_functions(file_path)
                        for func in functions:
                            # Create function node
                            session.run(
                                """
                                MERGE (fn:Function {
                                    name: $name,
                                    file_path: $file_path
                                })
                                SET fn.args = $args,
                                    fn.docstring = $docstring,
                                    fn.body = $body
                                """,
                                name=func["name"],
                                file_path=file_path,
                                args=",".join(func["args"]),
                                docstring=func["docstring"],
                                body=func["body"]
                            )
                            
                            # Create relationship between file and function
                            session.run(
                                """
                                MATCH (f:File {path: $file_path})
                                MATCH (fn:Function {name: $func_name, file_path: $file_path})
                                MERGE (f)-[:DEFINES]->(fn)
                                """,
                                file_path=file_path,
                                func_name=func["name"]
                            )
        
        logger.info(f"Successfully processed directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to process directory: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process directory")

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a text using the model."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def find_similar_functions(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Find similar functions based on query using embeddings."""
    query_embedding = generate_embedding(query)
    
    with driver.session() as session:
        functions = session.run("""
            MATCH (f:Function)
            RETURN f.name, f.body, f.docstring, f.file_path
        """)
        
        similarities = []
        for record in functions:
            func_text = f"{record['f.name']} {record['f.docstring']} {record['f.body']}"
            func_embedding = generate_embedding(func_text)
            
            similarity = cosine_similarity([query_embedding], [func_embedding])[0][0]
            
            similarities.append({
                'name': record['f.name'],
                'body': record['f.body'],
                'description': record['f.docstring'],
                'similarity_score': float(similarity),
                'file_path': record['f.file_path']
            })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.post("/process-repo", response_model=ProcessingStatus)
async def process_repo_endpoint(repo_request: RepoRequest, background_tasks: BackgroundTasks):
    """Process a GitHub repository."""
    try:
        # repo_dir = clone_repo(repo_request.repo_url)
        repo_dir = "./cloned_repo_20250211-135444"
        background_tasks.add_task(process_directory, repo_dir, driver)
        return ProcessingStatus(
            status="processing",
            message="Repository processing started"
        )
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[FunctionResponse])
async def search_functions_endpoint(query: QueryRequest):
    """Search for similar functions."""
    try:
        matches = find_similar_functions(query.query)
        return [
            FunctionResponse(
                name=match['name'],
                description=match['description'],
                body=match['body'],
                similarity_score=match['similarity_score'],
                file_path=match['file_path']
            )
            for match in matches
        ]
    except Exception as e:
        logger.error(f"Error searching functions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: QueryRequest):
    """Chat about code with context."""
    try:
        relevant_functions = find_similar_functions(query.query, top_k=1)
        
        template = """You are a helpful coding assistant. Use the following function as context to answer the question:
        
        Function:
        {context}
        
        Question: {query}
        
        Answer the question based on the function context and your general knowledge."""
        
        context = relevant_functions[0]['body'] if relevant_functions else "No relevant function found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{query}")
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"query": query.query, "context": context})
        
        return ChatResponse(
            response=response["text"],
            context={"relevant_function": relevant_functions[0] if relevant_functions else None}
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)