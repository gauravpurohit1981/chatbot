import os
from typing import Optional, List, Dict, Any
from langchain_community.vectorstores import FAISS

# Create a local response generator using a direct question-answering approach
class LocalChatEngine:
    """A local chat engine that uses retrieved documents to generate answers."""
    
    @staticmethod
    def format_docs(docs):
        """Format documents into a single string."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "Unknown source")
            # Include source information with the content
            formatted_docs.append(f"From {source}:\n{content}")
        return "\n\n---\n\n".join(formatted_docs)
    
    @staticmethod
    def extract_relevant_sentences(query: str, context: str) -> str:
        """Extract the most relevant sentences from the context based on keyword matching."""
        # Simplify the query and context for better matching
        query_words = set(query.lower().split())
        sentences = context.split('.')
        
        # Find sentences with the most query words
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count keyword matches
            sentence_words = set(sentence.lower().split())
            matching_words = query_words.intersection(sentence_words)
            
            if len(matching_words) > 0:
                relevant_sentences.append((sentence, len(matching_words)))
        
        # Sort by relevance (number of matching keywords)
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most relevant sentences
        result = '. '.join([s[0] for s in relevant_sentences[:3]])
        return result if result else context.split('.')[0]
    
    @staticmethod
    def generate_answer(query: str, context: str) -> str:
        """Generate an answer based on the query and context.
        
        This method processes the retrieved context and attempts to create a direct
        answer based on the most relevant information in the documents.
        """
        # First try to extract the most relevant sentences
        direct_answer = LocalChatEngine.extract_relevant_sentences(query, context)
        
        # Structure the response
        if direct_answer:
            response = f"Answer: {direct_answer}.\n\nThis information comes from your documents. Here are the relevant sections:\n\n{context}"
        else:
            response = f"I couldn't find specific information about '{query}' in your documents. Here's what I found that might be related:\n\n{context}"
        
        # Check if context is very short, which might indicate limited information
        if len(context.split()) < 30:
            response += "\n\nThe available documents contain limited information on this topic. Please upload more relevant documents for better answers."
        
        return response

def get_chat_response(query: str, vector_store: FAISS) -> str:
    """
    Generate a chat response based on the user query and document knowledge.
    
    Args:
        query: User query string.
        vector_store: FAISS vector store containing document embeddings.
        
    Returns:
        str: Generated response from the chatbot.
    """
    try:
        # Preprocess the query to improve matching
        processed_query = query.strip()
        
        # Improve query with specific words/phrases extraction
        keywords = []
        for word in processed_query.split():
            if len(word) > 3:  # Only consider meaningful words
                keywords.append(word.lower())
        
        # Create more targeted search by using important keywords
        enhanced_query = " ".join(keywords) if keywords else processed_query
        
        # Try to get exact document matches first using a higher threshold
        try:
            # Use similarity search with high threshold for precision
            precise_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3,  # Small number of highly relevant documents
                    "score_threshold": 0.7  # High threshold for precision
                }
            )
            precise_docs = precise_retriever.get_relevant_documents(enhanced_query)
            
            # If we have precise matches, prioritize those
            if precise_docs:
                print(f"Found {len(precise_docs)} precise matches")
                docs = precise_docs
            else:
                # Fall back to MMR for diverse results
                retriever = vector_store.as_retriever(
                    search_type="mmr",  # Maximal Marginal Relevance
                    search_kwargs={
                        "k": 6,  # Increase number of documents for better context
                        "fetch_k": 12,  # Fetch more initial documents
                        "lambda_mult": 0.6,  # Favor relevance over diversity (higher value = more relevance)
                    }
                )
                docs = retriever.get_relevant_documents(processed_query)
                print(f"Using MMR with {len(docs)} results")
        except Exception as e:
            print(f"Advanced retrieval error: {e}. Falling back to basic similarity.")
            # Fall back to similarity search if advanced methods are not available
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,  # Moderate number of documents
                }
            )
            docs = retriever.get_relevant_documents(processed_query)
        
        if not docs:
            # If no results, try with a more relaxed threshold
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "score_threshold": None}
            )
            docs = retriever.get_relevant_documents(processed_query)
            
        if not docs:
            return "I couldn't find any relevant information in the documents you provided. Please try rephrasing your question or upload more relevant documents."
        
        # Process documents by relevance
        if len(docs) > 1:
            # Sort documents by potential relevance based on metadata
            # This prioritizes documents that are more likely to have the answer
            # based on document name match with query keywords
            query_words = set(query.lower().split())
            for doc in docs:
                doc.metadata["relevance_score"] = 0
                source = doc.metadata.get("source", "").lower()
                # Check if document name contains any query words
                for word in query_words:
                    if word in source and len(word) > 3:  # Only consider meaningful words
                        doc.metadata["relevance_score"] += 1
            
            # Sort documents by this custom relevance score
            docs = sorted(docs, key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        
        # Extract and format the content from the retrieved documents
        chat_engine = LocalChatEngine()
        context = chat_engine.format_docs(docs)
        
        # Generate an answer based on the context
        answer = chat_engine.generate_answer(query, context)
        
        # Add source information with more detail
        if docs:
            sources = []
            for doc in docs:
                if "source" in doc.metadata:
                    source = doc.metadata["source"]
                    # Add page number if available
                    if "page" in doc.metadata:
                        source += f" (page {doc.metadata['page']})"
                    sources.append(source)
            
            if sources:
                unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
                answer += "\n\nSources: " + ", ".join(unique_sources[:3])  # Limit to top 3 sources
        
        return answer
    
    except Exception as e:
        print(f"Error in get_chat_response: {e}")
        return f"An error occurred while generating a response: {str(e)}"
