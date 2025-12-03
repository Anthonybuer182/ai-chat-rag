import os
import uuid
import sqlite3
import re
import logging
import requests
import concurrent.futures
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from typing import List, Optional, Callable
import chromadb
from chromadb.config import Settings
from text_chunk import recursive_text_split
from utils.stream_llm import stream_llm
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 初始化应用
app = FastAPI(title="RAG Demo")
logger.info("FastAPI应用初始化完成")

# 创建必要的目录
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)
logger.info("创建必要的目录: static/uploads, data")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
logger.info("静态文件挂载完成")

# 阿里百炼嵌入模型配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_EMBEDDING_API_URL = os.getenv("DASHSCOPE_EMBEDDING_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
DASHSCOPE_RERANK_API_URL = os.getenv("DASHSCOPE_RERANK_API_URL", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")

# 自定义嵌入函数（使用阿里百炼text-embedding-v4）
def batch_embedding_api_call(batch_texts, max_retries=3):
    """单批次嵌入API调用，包含重试机制"""
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-v4",
                "input": batch_texts
            }
            
            response = requests.post(f"{DASHSCOPE_EMBEDDING_API_URL}/compatible-mode/v1/embeddings", 
                                   headers=headers, 
                                   json=data,
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "data" not in result:
                    raise Exception("嵌入API返回格式异常")
                
                embeddings = [item["embedding"] for item in result["data"]]
                logger.info(f"批次嵌入API调用成功: 文本数={len(batch_texts)}, 嵌入维度={len(embeddings[0]) if embeddings else 0}")
                return embeddings
            else:
                error_msg = f"嵌入API调用失败: {response.status_code} - {response.text}"
                logger.warning(f"批次嵌入API调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 如果是参数错误（如批量大小超限），立即失败不重试
                if "batch size is invalid" in response.text:
                    raise Exception(error_msg)
                
                # 等待后重试
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # 指数退避
        
        except requests.exceptions.Timeout:
            logger.warning(f"批次嵌入API请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                raise Exception("嵌入API请求超时")
            import time
            time.sleep(2 ** attempt)
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            import time
            time.sleep(2 ** attempt)
    
    raise Exception("批次嵌入API调用达到最大重试次数")

def custom_embedding_function(texts, max_batch_size=10, max_workers=5):
    """使用阿里百炼text-embedding-v4生成文本嵌入，支持分批处理和多线程"""
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY环境变量未设置")
    
    # 将文本列表转换为字符串列表
    if isinstance(texts, str):
        texts = [texts]
    
    # 检查文本是否为空
    if not texts:
        return []
    
    # 如果文本数量小于等于批量大小，直接调用
    if len(texts) <= max_batch_size:
        return batch_embedding_api_call(texts)
    
    logger.info(f"开始分批处理嵌入: 总文本数={len(texts)}, 批量大小={max_batch_size}, 最大线程数={max_workers}")
    
    # 分批处理
    batches = []
    for i in range(0, len(texts), max_batch_size):
        batch = texts[i:i + max_batch_size]
        batches.append(batch)
    
    all_embeddings = []
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次任务
        future_to_batch = {executor.submit(batch_embedding_api_call, batch): batch for batch in batches}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                embeddings = future.result()
                all_embeddings.extend(embeddings)
                logger.info(f"批次处理完成: 文本数={len(batch)}, 累计嵌入数={len(all_embeddings)}")
            except Exception as e:
                logger.error(f"批次处理失败: 文本数={len(batch)}, 错误={e}")
                raise Exception(f"嵌入生成过程中发生错误: {e}")
    
    logger.info(f"阿里百炼嵌入生成完成: 总文本数={len(texts)}, 嵌入维度={len(all_embeddings[0]) if all_embeddings else 0}")
    return all_embeddings

# 初始化向量数据库客户端，使用自定义嵌入函数
chroma_client = chromadb.PersistentClient(
    path="data/chroma",
    settings=Settings(anonymized_telemetry=False)  # 禁用遥测
)
logger.info("向量数据库客户端初始化完成")
# 阿里百炼重排模型配置（使用gte-rerank-v2）
def rerank_results(query, retrieved_docs, top_k=5):
    """使用阿里百炼gte-rerank-v2进行重排"""
    if not retrieved_docs:
        return []
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY环境变量未设置")
    
    # 准备文档列表
    documents = [doc["text"] for doc in retrieved_docs]
    
    # 调用阿里百炼重排API
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gte-rerank-v2",
        "input": {
            "query": query,
            "documents": documents
        },
        "parameters": {
            "return_documents": True,
            "top_n": top_k
        }
    }
    
    response = requests.post(f"{DASHSCOPE_RERANK_API_URL}/api/v1/services/rerank/text-rerank/text-rerank",
                           headers=headers,
                           json=data,
                           timeout=30)
    
    if response.status_code != 200:
        logger.error(f"阿里百炼重排API调用失败: {response.status_code} - {response.text}")
        # 如果重排失败，返回原始结果的前top_k个
        logger.warning("重排API调用失败，返回原始检索结果")
        return retrieved_docs[:top_k]
    
    result = response.json()
    
    # 处理重排结果 - 根据gte-rerank-v2的实际返回格式
    if "output" not in result or "results" not in result["output"]:
        logger.warning("重排API返回结果格式异常，返回原始检索结果")
        return retrieved_docs[:top_k]
    
    reranked_results = result["output"]["results"]
    
    # 将重排结果与原始文档关联
    reranked_docs = []
    for rerank_item in reranked_results:
        original_index = rerank_item["index"]
        if original_index < len(retrieved_docs):
            original_doc = retrieved_docs[original_index]
            # 更新文档文本为重排后的文本（如果API返回了重排后的文本）
            if "document" in rerank_item and "text" in rerank_item["document"]:
                original_doc["text"] = rerank_item["document"]["text"]
            original_doc["rerank_score"] = rerank_item["relevance_score"]
            reranked_docs.append(original_doc)
    
    logger.info(f"阿里百炼重排完成: 查询='{query}', 输入文档数={len(retrieved_docs)}, 输出文档数={len(reranked_docs)}")
    return reranked_docs

# 初始化SQLite数据库
def init_db():
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    # # 删除旧表（如果存在）
    # c.execute('DROP TABLE IF EXISTS documents')
    # 创建新表（如果不存在）
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  original_filename TEXT,
                  uploaded_at TIMESTAMP)''')
    conn.commit()
    conn.close()
    logger.info("SQLite数据库初始化完成")

init_db()

# 添加文档记录
def add_document(filename, original_filename):
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    doc_id = str(uuid.uuid4())
    c.execute("INSERT INTO documents (id, filename, original_filename, uploaded_at) VALUES (?, ?, ?, ?)",
              (doc_id, filename, original_filename, datetime.now()))
    conn.commit()
    conn.close()
    logger.info(f"添加文档记录: ID={doc_id}, 文件名={original_filename}, 存储名={filename}")
    return doc_id

# 获取所有文档
def get_documents():
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
    documents = [dict(row) for row in c.fetchall()]
    conn.close()
    return documents

# 获取单个文档信息
def get_document(doc_id):
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    document = dict(result) if result else None
    conn.close()
    return document

# 检查文档是否已存在（通过原始文件名）
def document_exists(original_filename):
    conn = sqlite3.connect('data/knowledge_base.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE original_filename = ?", (original_filename,))
    result = c.fetchone()
    document = dict(result) if result else None
    conn.close()
    return document is not None

# 删除文档
def delete_document(doc_id):
    conn = sqlite3.connect('data/knowledge_base.db')
    c = conn.cursor()
    
    # 获取文件名
    c.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
    result = c.fetchone()
    if result:
        filename = result[0]
        logger.info(f"开始删除文档: ID={doc_id}, 文件名={filename}")
        
        # 删除文件
        file_path = f"static/uploads/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除文件: {file_path}")
        else:
            logger.warning(f"文件不存在: {file_path}")
        
        # 删除向量数据库中的集合
        try:
            collection_name = f"doc_{doc_id}"
            chroma_client.delete_collection(collection_name)
            logger.info(f"已删除向量集合: {collection_name}")
        except Exception as e:
            logger.error(f"删除向量集合时出错: {e}")
        
        # 删除数据库记录
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        logger.info(f"已删除数据库记录: 文档ID={doc_id}")
    
    conn.close()
    logger.info(f"删除操作结果: {'成功' if result is not None else '失败'}")
    return result is not None

# 存储文档到向量数据库
def store_document_in_vector_db(doc_id, text):
    # 获取或创建集合
    collection_name = f"doc_{doc_id}"
    try:
        collection = chroma_client.get_collection(collection_name)
        logger.info(f"获取现有向量集合: {collection_name}")
    except:
        collection = chroma_client.create_collection(
            collection_name,
            embedding_function=custom_embedding_function
        )
        logger.info(f"创建新的向量集合: {collection_name}")
    
    # 分块处理文本
    chunks = recursive_text_split(
        text=text,
        chunk_size=2024,
        chunk_overlap=0,
        separators=["**********","\r\n\r\n", "\n\n", "\r\n", "\n", ". ", "? ", "! ", " "]
    )
    logger.info(f"文档分块完成: 文档ID={doc_id}, 块数={len(chunks)}")
    
    # 生成嵌入（使用阿里百炼text-embedding-v4）
    embeddings = custom_embedding_function(chunks)
    logger.info(f"嵌入生成完成: 文档ID={doc_id}, 嵌入维度={len(embeddings[0]) if embeddings else 0}")
    
    # 准备元数据
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    
    # 添加到集合
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks,
        ids=ids
    )
    logger.info(f"文档存储到向量数据库完成: 文档ID={doc_id}, 总块数={len(chunks)}")

# 向量召回检索（使用自定义嵌入函数，避免ChromaDB自动下载模型）
def multi_retrieval(query, doc_ids, top_k=5):
    results = []
    
    for doc_id in doc_ids:
        try:
            collection = chroma_client.get_collection(f"doc_{doc_id}")
            
            # 仅使用基于嵌入相似度的检索，避免文本检索触发模型下载
            query_embedding = custom_embedding_function([query])
            embedding_results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            # 处理嵌入检索结果
            if embedding_results['documents']:
                for i, doc_list in enumerate(embedding_results['documents']):
                    for j, doc in enumerate(doc_list):
                        chunk_id = embedding_results['ids'][i][j]
                        results.append({
                            "text": doc,
                            "score": 1 - embedding_results['distances'][i][j],  # 转换为相似度分数
                            "source": f"嵌入相似度 (文档: {doc_id})"
                        })
            
        except Exception as e:
            logger.error(f"检索文档 {doc_id} 时出错: {e}")
            continue
    
    return results



# 首页路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 添加文档API
@app.post("/api/documents")
async def add_document_api(file: UploadFile = File(...)):
    logger.info(f"开始处理文件上传: 文件名={file.filename}, 文件大小={file.size}")
    
    # 检查文档是否已存在
    if document_exists(file.filename):
        logger.warning(f"文档已存在，拒绝重复上传: {file.filename}")
        return JSONResponse(
            content={
                "status": "error", 
                "message": f"文档 '{file.filename}' 已上传过，不能重复上传"
            },
            status_code=400
        )
    
    # 保存文件
    file_ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = f"static/uploads/{filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info(f"文件保存成功: 存储路径={file_path}")
    
    # 添加到数据库
    doc_id = add_document(filename, file.filename)
    
    # 读取文件内容 (这里简化处理，实际应该根据文件类型解析)
    try:
        text = content.decode('utf-8')
        logger.info(f"文件内容解码成功: 文档ID={doc_id}, 文本长度={len(text)}")
    except:
        text = str(content)
        logger.warning(f"文件内容无法解码为UTF-8, 使用字符串表示: 文档ID={doc_id}")
    
    # 存储到向量数据库
    store_document_in_vector_db(doc_id, text)
    logger.info(f"文档处理完成: 文档ID={doc_id}, 原始文件名={file.filename}")
    
    return JSONResponse(content={"status": "success", "doc_id": doc_id})

# 获取文档列表API
@app.get("/api/documents")
async def get_documents_api():
    documents = get_documents()
    return JSONResponse(content=documents)

# 下载文档API
@app.get("/api/documents/{doc_id}/download")
async def download_document(doc_id: str):
    document = get_document(doc_id)
    if not document:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)
    
    file_path = f"static/uploads/{document['filename']}"
    if not os.path.exists(file_path):
        return JSONResponse(content={"status": "error", "message": "文件不存在"}, status_code=404)
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=document['original_filename'],
        media_type='application/octet-stream'
    )

# 查看文档内容API
@app.get("/api/documents/{doc_id}/view")
async def view_document(doc_id: str):
    document = get_document(doc_id)
    if not document:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)
    
    file_path = f"static/uploads/{document['filename']}"
    if not os.path.exists(file_path):
        return JSONResponse(content={"status": "error", "message": "文件不存在"}, status_code=404)
    
    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码或返回二进制数据
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            # 如果还是失败，返回错误信息
            return JSONResponse(content={"status": "error", "message": "无法读取文件内容（不支持的编码格式）"}, status_code=400)
    
    # 返回HTML页面显示文档内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{document['original_filename']} - 文档查看</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                padding: 30px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .header h1 {{
                color: #6e8efb;
                margin-bottom: 10px;
            }}
            .filename {{
                color: #6c757d;
                font-size: 1.1rem;
            }}
            .content {{
                line-height: 1.6;
                font-size: 16px;
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #fafbfc;
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #6e8efb;
                max-height: 70vh;
                overflow-y: auto;
            }}
            .content::-webkit-scrollbar {{
                width: 8px;
            }}
            .content::-webkit-scrollbar-track {{
                background: #f1f3f5;
                border-radius: 4px;
            }}
            .content::-webkit-scrollbar-thumb {{
                background: #6e8efb;
                border-radius: 4px;
            }}
            .back-btn {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 24px;
                background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            .back-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(110, 142, 251, 0.3);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-file-alt"></i> 文档内容查看</h1>
                <div class="filename">{document['original_filename']}</div>
            </div>
            <div class="content">{content}</div>
            <div style="text-align: center; margin-top: 30px;">
                <a href="javascript:window.close()" class="back-btn">关闭窗口</a>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# 删除文档API
@app.delete("/api/documents/{doc_id}")
async def delete_document_api(doc_id: str):
    success = delete_document(doc_id)
    if success:
        return JSONResponse(content={"status": "success", "message": "文档删除成功"})
    else:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)

# 搜索文档内容API
@app.post("/api/documents/{doc_id}/search")
async def search_document_api(doc_id: str, request: Request):
    # 检查文档是否存在
    document = get_document(doc_id)
    if not document:
        return JSONResponse(content={"status": "error", "message": "文档不存在"}, status_code=404)
    
    # 解析请求体
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(content={"status": "error", "message": "搜索关键词不能为空"}, status_code=400)
    except:
        return JSONResponse(content={"status": "error", "message": "无效的请求格式"}, status_code=400)
    
    try:
        # 使用向量召回检索
        retrieved_docs = multi_retrieval(query, [doc_id], top_k=10)
        logger.info(f"检索向量召回完成: 检索到 {len(retrieved_docs)} 条结果")
        
        # 重排检索结果
        reranked_docs = rerank_results(query, retrieved_docs, top_k=5)
        logger.info(f"重排完成: 保留 {len(reranked_docs)} 条最相关结果")
        
        # 转换格式以保持API兼容性
        search_results = []
        for doc in reranked_docs:
            search_results.append({
                "content": doc["text"],
                "score": doc.get("rerank_score", doc.get("score", 0.5)),
                "page": 1,  # 默认页码
                "metadata": {"source": doc.get("source", "")}
            })
        
        return JSONResponse(content={
            "status": "success",
            "data": search_results,
            "document_name": document['original_filename']
        })
        
    except Exception as e:
        logger.error(f"搜索文档时出错: {e}")
        return JSONResponse(content={
            "status": "error", 
            "message": f"搜索失败: {str(e)}"
        }, status_code=500)

# WebSocket聊天连接
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    try:
        while True:
            data = await websocket.receive_json()
            logger.debug(f"收到WebSocket消息: {data}")
            
            if data['type'] == 'message':
                message = data['message']
                selected_docs = data.get('selected_docs', [])
                logger.info(f"处理用户消息: 消息长度={len(message)}, 选择文档数={len(selected_docs)}")
                
                # 从选定的文档中检索相关内容
                context = ""
                context_documents = []
                if selected_docs:
                    # 获取文档信息
                    documents_info = []
                    for doc_id in selected_docs:
                        doc_info = get_document(doc_id)
                        if doc_info:
                            documents_info.append({
                                "id": doc_id,
                                "name": doc_info['original_filename']
                            })
                    
                    # 检索向量召回检索
                    retrieved_docs = multi_retrieval(message, selected_docs, top_k=10)
                    logger.info(f"检索向量召回检索完成: 检索到 {len(retrieved_docs)} 条结果")
                    
                    # 重排检索结果
                    reranked_docs = rerank_results(message, retrieved_docs, top_k=5)
                    logger.info(f"重排完成: 保留 {len(reranked_docs)} 条最相关结果")
                    
                    # 按文档分组相关信息
                    doc_relevant_content = {}
                    for doc in reranked_docs:
                        # 从source中提取文档ID
                        source_match = re.search(r'文档: ([^)]+)', doc.get('source', ''))
                        if source_match:
                            doc_id = source_match.group(1)
                            if doc_id not in doc_relevant_content:
                                doc_relevant_content[doc_id] = []
                            doc_relevant_content[doc_id].append(doc['text'])
                    
                    # 构建上下文文档信息
                    for doc_info in documents_info:
                        doc_id = doc_info['id']
                        relevant_content = doc_relevant_content.get(doc_id, [])
                        context_documents.append({
                            "document_id": doc_id,
                            "document_name": doc_info['name'],
                            "relevant_content": relevant_content
                        })
                    
                    # 构建LLM上下文
                    context = "\n\n".join([f"[来源: {doc['source']}]\n{doc['text']}" for doc in reranked_docs])
                    
                    # 发送结构化的上下文信息
                    await websocket.send_json({
                        "type": "context",
                        "context": {
                            "total_documents": len(selected_docs),
                            "total_relevant_info": len(reranked_docs),
                            "documents": context_documents
                        }
                    })
                
                # 使用OpenAI API流式生成回复
                messages = []
                if context:
                    messages.append({"role": "system", "content": f"基于以下上下文信息回答问题：\n\n{context}\n\n请根据上下文提供准确、相关的回答。"})
                messages.append({"role": "user", "content": message})
                
                # 发送开始响应消息
                await websocket.send_json({
                    "type": "response_start"
                })
                logger.info("开始生成AI回复")
                
                # 流式返回响应
                full_response = ""
                async for chunk in stream_llm(messages):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "response_chunk",
                        "chunk": chunk
                    })
                
                # 发送结束响应消息
                await websocket.send_json({
                    "type": "response_end",
                    "full_response": full_response
                })
                logger.info(f"AI回复生成完成: 回复长度={len(full_response)}")
                
    except WebSocketDisconnect:
        logger.info("客户端断开WebSocket连接")
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
        await websocket.close(code=1011)

# 启动应用
if __name__ == "__main__":
    import uvicorn
    logger.info("启动FastAPI应用服务器...")
    logger.info(f"服务器地址: http://0.0.0.0:8000")
    logger.info(f"API文档地址: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
