from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import ast
import torch
import git
import time
import logging
import numpy as np
from git import Repo
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Code Analysis API",
             description="API for analyzing GitHub repositories and chatting about code")

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
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5,
)

# Pydantic models
class RepoRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    query: str

class FunctionResponse(BaseModel):
    name: str
    description: str
    body: str
    similarity_score: float

class ChatResponse(BaseModel):
    response: str

class ProcessingStatus(BaseModel):
    status: str
    message: str
conversation_memory = ConversationBufferMemory()

# Helper functions
def clone_repo(repo_url: str, target_dir_prefix: str = "cloned_repo") -> str:
    """Clone a GitHub repository."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target_dir = f"{target_dir_prefix}_{timestamp}"
    logger.info(f"Cloning repository from {repo_url} into {target_dir}...")
    Repo.clone_from(repo_url, target_dir)
    return target_dir

def process_directory(directory: str, driver: GraphDatabase.driver):
    """Process directory and store in Neo4j."""
    def create_function_node(tx, name, args, docstring, file_path, body):
        query = """
        CREATE (func:Function {
            name: $name, 
            args: $args, 
            docstring: $docstring, 
            file_path: $file_path, 
            body: $body
        })
        """
        tx.run(query, name=name, args=args, docstring=docstring, 
               file_path=file_path, body=body)

    def extract_functions(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=file_path)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node),
                        "body": ast.get_source_segment(content, node)
                    })
            return functions

    with driver.session() as session:
        for root, dirs, files in os.walk(directory):
            # Filter out unwanted directories and files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if f.endswith('.py') and not f.startswith('.')]
            
            for file in files:
                file_path = os.path.join(root, file)
                functions = extract_functions(file_path)
                
                for func in functions:
                    session.execute_write(
                        create_function_node,
                        func["name"],
                        ",".join(func["args"]),
                        func["docstring"] or "No docstring",
                        file_path,
                        func["body"]
                    )

def generate_embeddings(driver: GraphDatabase.driver):
    """Generate and store embeddings for function descriptions."""
    with driver.session() as session:
        # Fetch all functions
        functions = session.run("""
        MATCH (f:Function)
        RETURN f.name, f.body, f.docstring
        """)
        
        for record in functions:
            # Generate embedding from body and docstring
            text = f"{record['f.body']} {record['f.docstring']}"
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Store embedding
            session.run("""
            MATCH (f:Function {name: $name})
            SET f.embedding = $embedding
            """, name=record['f.name'], embedding=embedding.tolist())

def find_similar_functions(query: str, driver: GraphDatabase.driver, top_k: int = 3):
    """Find similar functions based on query."""
    # Generate query embedding
    tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()

    # Fetch functions with embeddings
    with driver.session() as session:
        functions = session.run("""
        MATCH (f:Function)
        WHERE exists(f.embedding)
        RETURN f.name, f.description, f.body, f.embedding
        """)
        
        similarities = []
        for record in functions:
            similarity = cosine_similarity(
                [query_embedding], 
                [np.array(record['f.embedding'])]
            )[0][0]
            similarities.append({
                'name': record['f.name'],
                'description': record.get('f.description', ''),
                'body': record['f.body'],
                'similarity_score': float(similarity)
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# API endpoints
@app.post("/process-repo", response_model=ProcessingStatus)
async def process_repo(repo_request: RepoRequest, background_tasks: BackgroundTasks):
    """
    Process a GitHub repository and store its code structure in Neo4j.
    """
    try:
        # Clone repository
        repo_dir = clone_repo(repo_request.repo_url)
        
        # Add processing tasks to background
        background_tasks.add_task(process_directory, repo_dir, driver)
        background_tasks.add_task(generate_embeddings, driver)
        
        return ProcessingStatus(
            status="processing",
            message=f"Repository cloning completed. Processing started in background."
        )
    
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[FunctionResponse])
async def search_functions(query: QueryRequest):
    """
    Search for similar functions based on the query.
    """
    try:
        matches = find_similar_functions(query.query, driver)
        return [
            FunctionResponse(
                name=match['name'],
                description=match['description'],
                body=match['body'],
                similarity_score=match['similarity_score']
            )
            for match in matches
        ]
    
    except Exception as e:
        logger.error(f"Error searching functions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(query: QueryRequest):
    """
    Chat about code using the LLM with conversation memory.
    """
    try:
        # Create conversation chain with memory
        conversation = ConversationChain(
            llm=llm,
            memory=conversation_memory,
            verbose=True
        )
        
        # Generate response using the conversation chain
        response = conversation.run(query.query)
        
        return ChatResponse(response=response)
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    driver.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)