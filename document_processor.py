import os
import uuid
import json
import pickle
import numpy as np
from typing import Optional, List, Dict, Any
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings

# Create a custom embedding class using scikit-learn
class TfidfEmbeddings(Embeddings):
    def __init__(self, max_features=8000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF embeddings with improved parameters.
        
        Args:
            max_features: Maximum number of features to include in the vocabulary.
            ngram_range: The range of n-values for different n-grams to be extracted.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,  # Use unigrams and bigrams for better semantic capture
            use_idf=True,             # Use inverse document frequency
            min_df=2,                 # Minimum document frequency for a term
            max_df=0.95,              # Maximum document frequency for a term (ignore very common words)
            sublinear_tf=True,        # Apply sublinear scaling to term frequencies
            strip_accents='unicode',  # Remove accents from characters
            analyzer='word',          # Analyze at word level
            stop_words='english'      # Remove English stop words
        )
        self.fitted = False
        
    def fit(self, texts):
        """Fit the vectorizer on a list of texts."""
        if not texts:
            # Handle empty text list
            return
        self.vectorizer.fit(texts)
        self.fitted = True
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using TF-IDF."""
        if not self.fitted:
            self.fit(texts)
            
        if not texts:
            # Handle empty text list
            return [[0.0] * self.vectorizer.max_features]
            
        vectors = self.vectorizer.transform(texts).toarray()
        # Normalize vectors to unit length for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        normalized_vectors = vectors / np.maximum(norms, 1e-10)
        return normalized_vectors.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text using TF-IDF."""
        if not self.fitted:
            # If not fitted, embed with zeros
            return [0.0] * self.vectorizer.max_features
            
        if not text.strip():
            # Handle empty query
            return [0.0] * self.vectorizer.max_features
            
        vector = self.vectorizer.transform([text]).toarray()[0]
        # Normalize vector to unit length
        norm = np.linalg.norm(vector)
        # Avoid division by zero
        normalized_vector = vector / max(norm, 1e-10)
        return normalized_vector.tolist()

# Constants
DOCUMENTS_DIR = "documents"
VECTOR_STORE_PATH = "faiss_index"
DOCUMENT_METADATA_FILE = "document_metadata.json"

# Create necessary directories
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

def get_document_ids() -> List[str]:
    """Get list of document IDs from metadata file."""
    if os.path.exists(DOCUMENT_METADATA_FILE):
        try:
            with open(DOCUMENT_METADATA_FILE, "r") as f:
                metadata = json.load(f)
                return list(metadata.keys())
        except Exception:
            return []
    return []

def get_document_metadata() -> Dict[str, Dict[str, Any]]:
    """Get document metadata from file."""
    if os.path.exists(DOCUMENT_METADATA_FILE):
        try:
            with open(DOCUMENT_METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_document_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    """Save document metadata to file."""
    with open(DOCUMENT_METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def get_loader_for_file(file_path: str):
    """Return the appropriate document loader based on file extension."""
    file_extension = file_path.split(".")[-1].lower()
    
    if file_extension == "pdf":
        return PyPDFLoader(file_path)
    elif file_extension in ["docx", "doc"]:
        return Docx2txtLoader(file_path)
    elif file_extension in ["txt", "md", "html", "htm"]:
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def process_document(file_path: str, original_filename: str, vector_store: Optional[FAISS] = None) -> bool:
    """
    Process a document and add it to the vector store.
    
    Args:
        file_path: Path to the document file.
        original_filename: Original name of the uploaded file.
        vector_store: Optional existing vector store to add documents to.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Generate document ID
        doc_id = f"{original_filename}_{uuid.uuid4().hex[:8]}"
        
        # Load the document
        loader = get_loader_for_file(file_path)
        documents = loader.load()
        
        # Use a more sophisticated text splitter for better context preservation
        # Smaller chunks with more overlap for specific answers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more specific information
            chunk_overlap=150,  # Significant overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]  # More granular splitting
        )
        
        # Enhanced metadata for better search and attribution
        for doc in documents:
            doc.metadata["source"] = original_filename
            doc.metadata["doc_id"] = doc_id
            # Add page numbers if available
            if "page" in doc.metadata:
                page_num = doc.metadata["page"]
                doc.metadata["source"] = f"{original_filename} (page {page_num})"
        
        # Split documents into chunks
        splits = text_splitter.split_documents(documents)
        
        # Add chunk position information to help with retrieval context
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["total_chunks"] = len(splits)
        
        if not splits:
            return False
        
        # Get or create vector store
        if vector_store is None:
            # Extract texts for fitting the vectorizer
            texts = [doc.page_content for doc in splits]
            
            # Create the TF-IDF embeddings
            embeddings = TfidfEmbeddings()
            embeddings.fit(texts)  # Explicitly fit the vectorizer
            
            # Create vector store
            vector_store = FAISS.from_documents(splits, embeddings)
            
            # Save the vector store
            vector_store.save_local(VECTOR_STORE_PATH)
            
            # Save the embeddings model
            with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        else:
            # Load the saved embeddings
            try:
                with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "rb") as f:
                    embeddings = pickle.load(f)
            except:
                # If embeddings don't exist, create new ones
                texts = [doc.page_content for doc in splits]
                embeddings = TfidfEmbeddings()
                embeddings.fit(texts)
            
            # Add to existing vector store
            vector_store.add_documents(splits)
            
            # Save the updated vector store
            vector_store.save_local(VECTOR_STORE_PATH)
            
            # Save the updated embeddings
            with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        # Update document metadata
        metadata = get_document_metadata()
        metadata[doc_id] = {
            "original_filename": original_filename,
            "date_added": str(uuid.uuid1()),
            "chunks": len(splits),
        }
        save_document_metadata(metadata)
        
        return True
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return False

def delete_document(doc_id: str, vector_store: Optional[FAISS] = None) -> bool:
    """
    Delete a document from the vector store.
    
    Args:
        doc_id: ID of the document to delete.
        vector_store: Optional existing vector store.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Update document metadata
        metadata = get_document_metadata()
        if doc_id in metadata:
            del metadata[doc_id]
            save_document_metadata(metadata)
        
        # Recreate vector store without the deleted document
        if vector_store is not None:
            new_docs = []
            for doc_id_check in metadata.keys():
                # Create search filter for documents with this ID
                search_filter = lambda metadata: metadata["doc_id"] != doc_id
                
                # Apply the filter when searching
                vector_store.delete(filter=search_filter)
            
            # Save the updated vector store
            vector_store.save_local(VECTOR_STORE_PATH)
            
            return True
        
        return False
    
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        return False

def load_existing_documents() -> Optional[FAISS]:
    """
    Load existing documents from FAISS vector store or rebuild if necessary.
    
    Returns:
        FAISS: The loaded vector store or None if it doesn't exist and can't be rebuilt.
    """
    try:
        # Check if both the index and embeddings exist
        index_exists = os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss")
        embeddings_exist = os.path.exists(f"{VECTOR_STORE_PATH}/embeddings.pkl")
        
        if os.path.exists(VECTOR_STORE_PATH) and index_exists and embeddings_exist:
            try:
                # Load the saved embeddings
                with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "rb") as f:
                    embeddings = pickle.load(f)
                    
                # Load the vector store with allow_dangerous_deserialization=True 
                # since this is our own local store created within the app
                vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Quick validation check
                try:
                    test_query = "test query for validation"
                    vector_store.similarity_search(test_query, k=1)
                    print("Vector store validation successful")
                    return vector_store
                except Exception as e:
                    print(f"Vector store validation failed: {e}. Will rebuild.")
                    # Continue to rebuild if validation fails
            except Exception as e:
                print(f"Error loading vector store: {e}. Will attempt to rebuild.")
                # Continue to rebuild if loading fails
        
        # If we reach here, we need to create/rebuild the vector store
        print("Creating new vector store or rebuilding existing one.")
        
        # Get document metadata to rebuild from
        metadata = get_document_metadata()
        document_ids = list(metadata.keys())
            
        if document_ids:
            # Rebuild from existing documents
            print(f"Rebuilding index from {len(document_ids)} existing documents.")
            
            # Create a new embeddings model
            embeddings = TfidfEmbeddings()
            all_splits = []
            all_texts = []
            
            # Process each document in the metadata
            # Note: This assumes the original files are still in the documents directory
            for doc_id in document_ids:
                filename = metadata[doc_id]["original_filename"]
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                
                if os.path.exists(file_path):
                    try:
                        # Load and process the document
                        loader = get_loader_for_file(file_path)
                        docs = loader.load()
                        
                        # Apply the same splitting logic as in process_document
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=150,
                            length_function=len,
                            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
                        )
                        
                        for doc in docs:
                            doc.metadata["source"] = filename
                            doc.metadata["doc_id"] = doc_id
                            if "page" in doc.metadata:
                                page_num = doc.metadata["page"]
                                doc.metadata["source"] = f"{filename} (page {page_num})"
                        
                        splits = text_splitter.split_documents(docs)
                        
                        for i, split in enumerate(splits):
                            split.metadata["chunk_id"] = i
                            split.metadata["total_chunks"] = len(splits)
                        
                        all_splits.extend(splits)
                        all_texts.extend([split.page_content for split in splits])
                    except Exception as e:
                        print(f"Error rebuilding document {filename}: {e}")
            
            if all_splits:
                # Fit embeddings on all texts
                embeddings.fit(all_texts)
                
                # Create a new vector store
                vector_store = FAISS.from_documents(all_splits, embeddings)
                
                # Ensure the directory exists
                os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
                
                # Save the vector store
                vector_store.save_local(VECTOR_STORE_PATH)
                
                # Save the embeddings
                with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "wb") as f:
                    pickle.dump(embeddings, f)
                
                print(f"Successfully rebuilt vector store with {len(all_splits)} chunks.")
                return vector_store
        
        # If no documents or rebuilding failed, create an empty vector store
        print("Creating empty vector store.")
        embeddings = TfidfEmbeddings()
        
        # Create directory for vector store
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        # Simple placeholder text
        placeholder_text = "This is a placeholder document for the empty vector store."
        
        # Fit the embeddings on the placeholder text
        embeddings.fit([placeholder_text])
        
        # Create the vector store
        vector_store = FAISS.from_texts([placeholder_text], embeddings)
        
        # Save the vector store
        vector_store.save_local(VECTOR_STORE_PATH)
        
        # Save the embeddings
        with open(f"{VECTOR_STORE_PATH}/embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
            
        return vector_store
            
    except Exception as e:
        print(f"Critical error loading/rebuilding vector store: {str(e)}")
        return None
