from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os, ast, json, time, pickle, asyncio, logging
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Optional,
)

from redis import Redis
import nltk, redis, pinecone
from rich.live import Live
from rich.table import Table
from tqdm.asyncio import tqdm
from rich.console import Console
from tqdm import tqdm as sync_tqdm
from nltk.tokenize import word_tokenize
from langchain_redis.cache import RedisCache
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_redis import RedisSemanticCache
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from pinecone import (
    Pinecone,
    ServerlessSpec,
)
from langchain_core.runnables.config import run_in_executor
from langchain_pinecone.vectorstores import DistanceStrategy
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters.python import PythonCodeTextSplitter
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.retrievers import (
    BM25Retriever,
    PineconeHybridSearchRetriever,
)
from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
    DirectoryLoader,
)
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    ConfigurableField,
    RunnablePassthrough,
)
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from userbot.core.session import jarina

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add console handler for embedding responses
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('API Response: %(message)s')
console_handler.setFormatter(console_formatter)
api_logger = logging.getLogger("api_responses")
api_logger.addHandler(console_handler)
api_logger.setLevel(logging.INFO)

# Constants for embeddings and batching
EMBEDDING_DIMENSION = 3072
MAX_INPUT_LENGTH = 8192
MAX_BATCH_SIZE = 80  # Changed to 80 for optimal chunk size
REDIS_TTL = 10800  # 180 minutes
BM25_PARAMS = {"k1": 1.5, "b": 0.75}

REDIS_URL = "redis://172.16.238.155:6379"
AZURE_OPENAI_API_KEY = "989bb2fa71594e5d8b4b8ee0ab749d0e"
AZURE_OPENAI_ENDPOINT = "https://jarinachat.openai.azure.com"
PINECONE_ENV = "us-east-2"
PINECONE_INDEX = "tgsummaries"
PINECONE_NAMESPACE = "default"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class DynamicThrottler:
    def __init__(self):
        self.rate_limit = 30
        self.time_window = 60  # seconds
        self.requests = []
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            current_time = time.time()
            self.requests = [
                req_time for req_time in self.requests
                if current_time - req_time < self.time_window
            ]

            if len(self.requests) >= self.rate_limit:
                sleep_time = self.requests[0] + self.time_window - current_time
                await asyncio.sleep(sleep_time)

            self.requests.append(current_time)

from telethon import TelegramClient

class TelegramScraper:
    def __init__(self, client):
        self.client = client

    async def fetch_messages(self, chat_id, limit=100):
        messages = []
        async for message in self.client.iter_messages(chat_id, limit=limit):
            messages.append(message.text)
        return messages

from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text):
        return self.splitter.split_text(text)

import ast
from datetime import datetime

class MetadataExtractor:
    def extract_metadata(self, content, file_path):
        metadata = {
            "file_path": file_path,
            "language": "python",
            "classes": [],
            "functions": [],
            "imports": [],
            "size_bytes": len(content),
            "created_at": datetime.now().isoformat()
        }
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)
                elif isinstance(node, ast.Import):
                    metadata["imports"].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ""
                    metadata["imports"].extend(f"{module}.{alias.name}" for alias in node.names)
        except Exception as e:
            print(f"Metadata extraction failed for {file_path}: {e}")
        return metadata

import pinecone
from langchain_pinecone import PineconeVectorStore

class PineconeManager:
    def __init__(self, api_key, environment, index_name, namespace):
        self.index = pinecone.Index(index_name)
        self.vector_store = PineconeVectorStore(embedding=None, index=self.index, namespace=namespace)

    def upsert_vectors(self, vectors):
        self.index.upsert(vectors=vectors)

from langchain_openai import AzureOpenAIEmbeddings

class AzureOpenAIManager:
    def __init__(self, azure_deployment, dimensions):
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_deployment, dimensions=dimensions)

    async def embed_documents(self, texts):
        return await self.embeddings.aembed_documents(texts)

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)

class ProgressTracker:
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[bold blue]{task.fields[filename]}"),
            BarColumn(),
            TimeElapsedColumn(),
        )

    def start(self, total, description):
        self.task = self.progress.add_task(description, total=total, filename="")

    def update(self, filename):
        self.progress.update(self.task, filename=filename)

    def advance(self):
        self.progress.advance(self.task)

class LangChainTelegramPlugin:
    def __init__(self, client, azure_deployment, dimensions, pinecone_api_key, pinecone_environment, pinecone_index, pinecone_namespace):
        self.scraper = TelegramScraper(client)
        self.text_splitter = TextSplitter()
        self.metadata_extractor = MetadataExtractor()
        self.pinecone_manager = PineconeManager(pinecone_api_key, pinecone_environment, pinecone_index, pinecone_namespace)
        self.azure_openai_manager = AzureOpenAIManager(azure_deployment, dimensions)
        self.throttler = DynamicThrottler()
        self.progress_tracker = ProgressTracker()

    async def process_messages(self, chat_id):
        messages = await self.scraper.fetch_messages(chat_id)
        self.progress_tracker.start(len(messages), "Processing messages")

        for message in messages:
            self.progress_tracker.update(message[:20])
            chunks = self.text_splitter.split_text(message)
            metadata = self.metadata_extractor.extract_metadata(message, "telegram_message")

            for chunk in chunks:
                await self.throttler.wait()
                embeddings = await self.azure_openai_manager.embed_documents([chunk])
                vectors = [
                    {
                        "id": f"{metadata['file_path']}_{i}",
                        "values": embedding,
                        "metadata": metadata
                    }
                    for i, embedding in enumerate(embeddings)
                ]
                self.pinecone_manager.upsert_vectors(vectors)
            self.progress_tracker.advance()

@jarina.sess_cmd(
    pattern="d(own)?l(oad)?(?:\s|$)([\s\S]*)",
    command=("download", plugin_category),
    info={
        "header": "To download the replied telegram file",
        "description": "Will download the replied telegram file to server .",
        "note": "The downloaded files will auto delete if you restart heroku.",
        "usage": [
            "{tr}download <reply>",
            "{tr}dl <reply>",
            "{tr}download custom name<reply>",
        ],
    },
)
async def download_handler(event):
    azure_deployment = "your_azure_deployment"
    dimensions = 3072
    pinecone_api_key = "your_pinecone_api_key"
    pinecone_environment = "us-east-2"
    pinecone_index = "your_pinecone_index"
    pinecone_namespace = "default"

    plugin = LangChainTelegramPlugin(event.client, azure_deployment, dimensions, pinecone_api_key, pinecone_environment, pinecone_index, pinecone_namespace)
    await plugin.process_messages(event.chat_id)

# Additional code for logging vector store index creation
class UnifiedCodeEngine:
    def __init__(self, namespace=PINECONE_NAMESPACE):
        # Set namespace and initialize Pinecone index
        self.namespace = namespace
        pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-2")
        if PINECONE_INDEX not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIMENSION,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-2"),
                metadata_config={"indexed": ["file_path", "language", "chunk_index"]}
            )
            logger.info(f"Pinecone index {PINECONE_INDEX} created successfully.")
        else:
            logger.info(f"Pinecone index {PINECONE_INDEX} already exists.")
        self.index = pc.Index(PINECONE_INDEX)
        self.llm = AzureChatOpenAI(
            azure_deployment="jarina-4o",
            temperature=0.7,
            model_kwargs={
                "api_version": "2024-03-01-preview"
            }
        )
        self.state = self.State(messages=[], context=[])
        # Initialize embeddings, BM25 encoder, vector store, code splitter, and throttler
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment="jarina-vector", dimensions=EMBEDDING_DIMENSION)
        self.vector_store = PineconeVectorStore(embedding=self.embeddings, index=self.index, namespace=self.namespace)
        self.bm25_encoder = BM25Encoder.default()
        self.bm25_encoder.fit(["dummy content"])
        self.code_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
        self.throttler = DynamicThrottler()

    async def embed_batch(self, texts: List[str], config: RunnableConfig = None) -> List[List[float]]:
        """Process batch with caching and rate limiting"""
        embeddings = []
        for text in texts:
            # Apply throttling before each API call
            await self.throttler.wait()

            cache_key = hash(text)
            if cached := await self._get_cached_embedding(cache_key):
                embeddings.append(cached)
                self.stats["cached_hits"] += 1
            else:
                try:
                    # Generate embeddings with rate limiting
                    dense = await self.embeddings.aembed_query(text)
                    sparse = self.bm25_encoder.encode_documents([text])[0]
                    combined = (dense, sparse)
                    await self._cache_embedding(cache_key, combined)
                    embeddings.append(combined)
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")
                    continue

        return embeddings

    async def _init_bm25(self, documents: List[Document]):
        """Initialize BM25 retriever with documents"""
        self.bm25_retriever = await run_in_executor(
            None,
            BM25Retriever.from_documents,
            documents,
            preprocess_func=lambda x: x.split()
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upsert_vectors(self, vectors):
        await self.index.upsert(vectors=vectors, namespace=self.namespace)

    async def _get_cached_embedding(self, key: str) -> Optional[Tuple]:
        """Retrieve cached embedding with metadata"""
        if cached := await self.redis.get(f"embed:{key}"):
            return pickle.loads(cached.encode())
        return None

    async def _cache_embedding(self, key: str, embedding: Tuple):
        """Store embedding with TTL"""
        await self.redis.setex(f"embed:{key}", REDIS_TTL, pickle.dumps(embedding))

    def _tokenize_code(self, text: str) -> List[str]:
        """BM25-optimized tokenization for code"""
        return [
            token for token in ast.parse(text).body
            if isinstance(token, (ast.FunctionDef, ast.ClassDef, ast.Import))
        ]

    async def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return None

        logger.error(f"Failed to read {file_path} with any supported encoding")
        return None

    async def process_directory(self, directory_path: str, progress, task) -> List[Document]:
        """Process directory with batched processing"""
        await self.redis.flushdb()  # Clear cache
        documents = []

        py_files = list(Path(directory_path).rglob("*.py"))
        total_files = len(py_files)

        progress.update(task, total=total_files)

        for file_path in py_files:
            try:
                progress.update(task, description=f"[cyan]Processing {file_path.name}")

                # Use the new read_file_content method
                content = await self.read_file_content(file_path)
                if content is None:
                    continue

                metadata = self.extract_code_metadata(content, str(file_path))

                # Split content into chunks
                chunks = self.code_splitter.split_text(content)

                # Create chunk processing subtask
                chunk_task = progress.add_task(
                    f"[yellow]Chunks for {file_path.name}",
                    total=len(chunks)
                )

                # Process chunks in batches
                for i in range(0, len(chunks), MAX_BATCH_SIZE):
                    batch_chunks = chunks[i:i + MAX_BATCH_SIZE]
                    batch_docs = [
                        Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": i + idx,
                                "total_chunks": len(chunks)
                            }
                        )
                        for idx, chunk in enumerate(batch_chunks)
                    ]

                    # Process batch
                    success = await self._process_batch(batch_docs)
                    if success:
                        documents.extend(batch_docs)
                        progress.update(chunk_task, advance=len(batch_chunks))

                # Update main progress after file is complete
                progress.update(task, advance=1)

                # Log file stats
                logger.info(f"Processed {file_path}")
                logger.info(f"  Classes: {len(metadata['classes'])}")
                logger.info(f"  Functions: {len(metadata['functions'])}")
                logger.info(f"  Imports: {len(metadata['imports'])}")
                logger.info(f"  Chunks: {len(chunks)}")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return documents

    async def main():
        """Interactive prompt for code embedding/retrieval and RAG-based answering"""
        print("\nCode Embedding and Retrieval System")
        print("===================================")
        print("1. Embed code from directory")
        print("2. Search existing embeddings")
        print("3. Ask a question (RAG)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")
        directory2 = "./jarina/userbot/core" #input("Enter the path to load embeddings from: ")
        engine = UnifiedCodeEngine()  # Single instance handles both operations

        if choice == "1":
           
