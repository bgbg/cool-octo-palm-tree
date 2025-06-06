#!/usr/bin/env python
import datetime
import re
from typing import List, Dict, Optional, Literal, Any, Union, Tuple
import logging
import os
import sys
import uuid
import textwrap
from cachier import cachier
import defopt
import numpy as np
import pandas as pd
import dotenv
from sqlalchemy import text, bindparam
from tqdm import tqdm
import requests
from joblib import Parallel, delayed

from orm import get_db_engine, get_db_session, Chunk, Document, Embedding
from embedding import EmbeddingService, OpenAIEmbeddingService
from query_enhancer import QueryEnhancer

import hebrew_utils as hu

# Import OpenAI for answer generation
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangchainDocument

# Remove local definition, import from cost_logging.py
from cost_logging import log_llm_cost, OPENAI_LLM_PRICING

orininal_print = print
print = hu.hebrew_print

# Set up logging to a single file for all modules
log_path = os.path.abspath("takanot_rag.log")
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL", logging.INFO),
    format="%(asctime)s|%(name)s|%(levelname)s| %(message)s",
    datefmt="%Y%m%d%H%M%S",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
    force=True,
)
logger = logging.getLogger("takanot_rag")
logging.getLogger("search").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Load environment variables
dotenv.load_dotenv()

# Check for required environment variables
connection_string = os.getenv("SUPABASE_CONNECTION_STRING")
if not connection_string:
    logging.error("Missing SUPABASE_CONNECTION_STRING environment variable")
    sys.exit(1)

engine = get_db_engine(connection_string)
session = get_db_session(engine)
logging.debug("Successfully connected to the database")


_MODEL_TO_EMBEDDING_SERVICE = dict()


def weighted_mean_top_k(
    scores: pd.Series,
    k: int,
    ascending: bool,
    strategy: Literal["score-based", "uniform"] = "score-based",
) -> float:
    """
    Calculate a weighted mean of the top-k scores, giving more importance to higher-ranked items.

    This function sorts scores, takes the top-k values, and applies a linearly decreasing
    weight pattern to emphasize higher-scoring items. This is particularly useful for
    document ranking where the best-matching chunks should have more influence.

    Args:
        scores: Series of scores to compute weighted mean from
        k: Number of top scores to consider
        ascending: If True, lower values are better (distances); if False, higher values
                  are better (similarities)

    Returns:
        Weighted mean of the top-k scores
    """
    sorted_scores = scores.sort_values(ascending=ascending).head(k)
    n = len(sorted_scores)
    if strategy == "uniform":
        weights = np.linspace(1, 0.1, n)  # or use any decreasing pattern
    elif strategy == "score-based":
        if ascending:
            # higher scores are better
            weights = sorted_scores.values
        else:
            # lower scores are better
            weights = 1 / (sorted_scores.values + 1)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    weights /= weights.sum()  # normalize weights to sum to 1
    return np.dot(sorted_scores, weights)


# @cachier(stale_after=datetime.timedelta(days=1))
def vector_search(
    query: str,
    strategy: Literal[
        "distance_cosine",
        "distance_euclidean",
        "similarity_cosine",
        "similarity_dot",
    ] = "similarity_cosine",
    top_k: int = 5,
    context: int = 1,
) -> pd.DataFrame:
    """
    Search documents using vector similarity and display results.

    Args:
        query: Text query to search for
        top_k: Number of results to return
        model_name: Name of the embedding model to use
        strategy: Similarity strategy to use

    Similarity/Distance Measures:
        - "similarity_cosine":
            Recommended for most semantic embedding models;
            measures cosine of angle between vectors (higher is more similar).
        - "similarity_dot":
            Use with models where embedding magnitudes are meaningful,
            e.g. unnormalized transformer outputs.
        - "distance_cosine":
            Same as cosine similarity but as a distance metric (lower is better);
            used when APIs expect distance.
        - "distance_euclidean":
            Use only when embedding space is geometrically meaningful
            (e.g., traditional word2vec or tabular features).

    Choose cosine similarity for most use cases involving sentence embeddings;
    use dot product if you know your model is optimized for it.
    """
    embedding_service = OpenAIEmbeddingService()
    model_name = "OPENAI"
    model_version = "text-embedding-3-large"
    chunk_method_label = "max_characters=700, overlap=50"

    query_embedding = embedding_service.generate_embedding(query)
    embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

    # Select SQL expression and order based on strategy
    strategy_map = {
        "distance_cosine": ("embedding <=> CAST(:embedding_str AS vector)", "ASC"),
        "distance_euclidean": ("embedding <-> CAST(:embedding_str AS vector)", "ASC"),
        "similarity_cosine": (
            "1 - (embedding <=> CAST(:embedding_str AS vector))",
            "DESC",
        ),
        "similarity_dot": ("embedding <*> CAST(:embedding_str AS vector)", "DESC"),
    }
    if strategy not in strategy_map:
        raise ValueError(f"Invalid strategy: {strategy}")

    expression, order = strategy_map[strategy]

    sql = text(
        f"""
        SELECT
            c.chunk_id,
            c.document_id,
            d.title,
            d.storage_path,
            c.page_number,
            c.sequence_number,
            c.text,
            {expression} AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        JOIN embeddings e ON e.chunk_id = c.chunk_id
        WHERE e.model_name = :model_name
          AND e.model_version = :model_version
          AND c.chunk_method_label = :chunk_method_label
        ORDER BY score {order}
        LIMIT :top_k
        """
    ).bindparams(
        bindparam("model_name", value=model_name, literal_execute=True),
        bindparam(
            "model_version",
            value=model_version,
            literal_execute=True,
        ),
        bindparam("top_k", value=top_k, literal_execute=True),
        bindparam("embedding_str", value=embedding_str, literal_execute=True),
        bindparam("chunk_method_label", value=chunk_method_label, literal_execute=True),
    )

    compiled_sql = str(
        sql.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
    )
    logger.debug(f"Executing SQL: {compiled_sql}")

    with engine.connect() as conn:
        result = conn.execute(sql)

        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result.fetchall()]

        if not records:
            logger.warning("No results found for the query")
            return pd.DataFrame(
                columns=[
                    "chunk_id",
                    "document_id",
                    "title",
                    "storage_path",
                    "page_number",
                    "sequence_number",
                    "text",
                    "score",
                    "document_score",
                    "total_score",
                    "is_context",
                    "result_number",
                ]
            )

        ret = pd.DataFrame(records)

        # Determine sort direction and scoring type based on strategy
        ascending = order == "ASC"
        is_similarity = not ascending

        # Add document-level scores (mean of top-k chunks per document)
        ret["document_score"] = ret.groupby("document_id")["score"].transform(
            lambda x: weighted_mean_top_k(
                x,
                top_k,
                ascending,
                strategy="score-based",
            )
        )

        # Sort results by document score then individual chunk score
        ret = ret.sort_values(
            ["document_score", "score"], ascending=[ascending, ascending]
        ).reset_index(drop=True)

        # Compute total score (higher is always better)
        doc_weight = 2.0  # Document score has more weight
        chunk_weight = 1.0

        if is_similarity:
            # For similarity metrics, higher is better
            ret["total_score"] = (
                doc_weight * ret["document_score"] + chunk_weight * ret["score"]
            )
        else:
            # For distance metrics, lower is better, so we negate
            ret["total_score"] = (
                -1 * doc_weight * ret["document_score"] - chunk_weight * ret["score"]
            )

        # add running number to the results, we'll need it for the context
        ret["result_number"] = np.arange(len(ret))
        # Mark all current chunks as not context
        ret["is_context"] = False

        # If context is requested, fetch context chunks
        if context > 0 and not ret.empty:
            ret = add_context_chunks(
                results=ret,
                context=context,
                model_name=model_name,
                model_version=model_version,
                conn=conn,
            )

    # Log summary statistics
    logger.info(
        f"Found {len(ret[~ret['is_context']])} direct matches for query: '{query}'"
    )
    if context > 0:
        logger.info(f"Added {len(ret[ret['is_context']])} context chunks")
    logger.info(
        f"Top document: {ret[~ret['is_context']].iloc[0]['title'] if not ret[~ret['is_context']].empty else 'None'}"
    )

    ret = collapse_context_chunks(ret)

    return ret


def add_context_chunks(
    *, results: pd.DataFrame, context: int, conn, model_name: str, model_version: str
) -> pd.DataFrame:
    """
    Add context chunks before and after each matched chunk.

    Args:
        results: DataFrame with search results
        context: Number of context chunks to retrieve before and after each match
        conn: Database connection
        model_name: Embedding model name
        model_version: Embedding model version

    Returns:
        DataFrame with both search results and context chunks
    """
    # Get unique document_ids and their sequence_numbers for top matches
    doc_seq_pairs = (
        results[["document_id", "sequence_number", "result_number"]]
        .dropna()
        .drop_duplicates()
    )

    # Prepare to store context chunks
    context_chunks = []

    # For each result, get context chunks
    for _, row in doc_seq_pairs.iterrows():
        doc_id = row["document_id"]
        seq_num = int(row["sequence_number"])
        result_number = int(row["result_number"])

        # Calculate context range
        min_seq = max(1, seq_num - context)
        max_seq = seq_num + context

        # Query for context chunks
        context_sql = text(
            """
            SELECT
                c.chunk_id,
                c.document_id,
                d.title,
                d.storage_path,
                c.page_number,
                c.sequence_number,
                c.text,
                0 AS score,
                :result_number AS result_number
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN embeddings e ON e.chunk_id = c.chunk_id
            WHERE c.document_id = :doc_id
              AND c.sequence_number >= :min_seq
              AND c.sequence_number <= :max_seq
              AND c.sequence_number != :seq_num
              AND e.model_name = :model_name
              AND e.model_version = :model_version
            """
        ).bindparams(
            bindparam("doc_id", value=doc_id, literal_execute=True),
            bindparam("min_seq", value=min_seq, literal_execute=True),
            bindparam("max_seq", value=max_seq, literal_execute=True),
            bindparam("seq_num", value=seq_num, literal_execute=True),
            bindparam("model_name", value=model_name, literal_execute=True),
            bindparam("model_version", value=model_version, literal_execute=True),
            bindparam("result_number", value=result_number, literal_execute=True),
        )

        compiled_context_sql = str(
            context_sql.compile(
                dialect=engine.dialect, compile_kwargs={"literal_binds": True}
            )
        )
        logger.debug(f"Executing SQL: {compiled_context_sql}")

        context_result = conn.execute(context_sql)

        # Add context chunks to our list
        context_columns = context_result.keys()
        for ctx_row in context_result.fetchall():
            context_chunks.append(dict(zip(context_columns, ctx_row)))

    # If we found context chunks, add them to the results
    if context_chunks:
        # Create DataFrame for context chunks
        df_context = pd.DataFrame(context_chunks)

        if not df_context.empty:
            # Add required columns to match main results
            df_context["document_score"] = 0
            df_context["total_score"] = 0
            df_context["is_context"] = True

            # Combine main results with context chunks
            results = pd.concat([results, df_context], ignore_index=True)

            # Sort by document_id and sequence_number to keep context chunks
            # adjacent to their matching chunks
            results = results.sort_values(
                ["result_number", "sequence_number"]
            ).reset_index(drop=True)

    return results


def collapse_context_chunks(results: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse context chunks into a single row per result_number.
    """
    gr = results.groupby("result_number")
    combined_rows = []
    for result_number, result in gr:

        sel_not_context = ~result.is_context.values
        assert sel_not_context.sum() == 1

        combined_row = result[sel_not_context].iloc[0].copy()
        combined_text = "\n".join(result["text"].values)
        combined_row["text"] = combined_text
        combined_rows.append(combined_row)

    combined_df = (
        pd.DataFrame(combined_rows).reset_index(drop=True).drop(columns=["is_context"])
    )
    return combined_df


def display_results(
    results: pd.DataFrame, query: str, strategy: str, top_k: int, context: int
) -> None:
    """
    Display search results in a formatted way, with context chunks collapsed into the main result.

    This function first collapses context chunks into their parent results using
    collapse_context_chunks, then displays each result with its combined text.
    Each result's text is formatted with proper wrapping and indentation.

    Args:
        results: DataFrame with search results and context chunks
        query: The search query that was used
        strategy: The search strategy that was used (e.g., 'similarity_cosine')
        top_k: Number of top results that were requested
        context: Number of context chunks that were requested per result
    """
    if results.empty:
        print(f"No results found for query: '{query}'")
        return

    print(f"\nSearch results for: '{query}'")
    print(f"Strategy: {strategy}, Top-K: {top_k}, Context: {context}\n")

    for _, row in results.iterrows():
        # Start a new recument section if we've moved to a new document
        n_doc_matches = results.document_id.nunique()
        print(f"\n=== Document: {row['title']} ({n_doc_matches} matches) ===")
        text = row["text"]

        # Format text to maximum 120 columns
        wrapped_text = textwrap.fill(text, width=80, replace_whitespace=False)

        print(f"    Sequence: {row['sequence_number'] or 'N/A'}")
        print(f"    Chunk ID: {row['chunk_id']}")
        print(f"    Text:")
        print(textwrap.indent(wrapped_text, "        "))
        print()


def format_search_results(
    results: pd.DataFrame,
    query: str,
    strategy: str,
    top_k: int,
    enhancement_techniques: List[str],
) -> str:
    """
    Format enhanced search results into a readable string.

    Args:
        results: DataFrame with search results and context chunks
        query: The original search query that was used
        strategy: The search strategy that was used
        top_k: Number of top results that were requested per query
        enhancement_techniques: The enhancement techniques that were used

    Returns:
        A formatted string containing the search results
    """
    if results.empty:
        return f"No results found for query: '{query}'"

    output = []
    output.append(f"\n# Enhanced search results for: '{query}'")
    output.append(f"# Strategy: {strategy}, Top-K per query: {top_k}")
    output.append(
        "# Enhancement techniques: " + ",".join(enhancement_techniques) + "\n"
    )

    # Check if results have enhancement info
    has_enhancement_info = all(
        col in results.columns for col in ["source_query", "enhancement_type"]
    )

    # Get unique document IDs and their scores
    unique_docs = results[["document_id", "title", "document_score"]].drop_duplicates()
    unique_docs = unique_docs.sort_values("document_score", ascending=False)

    output.append(f"# Found {len(unique_docs)} relevant documents\n")

    # Group results by document for better display
    for doc_index, (_, doc_row) in enumerate(unique_docs.iterrows(), 1):
        doc_id = doc_row["document_id"]
        doc_title = doc_row["title"]

        # Get chunks for this document
        doc_chunks = results[results["document_id"] == doc_id]

        output.append(
            f"\n=== Document {doc_index}: {doc_title} ({len(doc_chunks)} relevant chunks) ==="
        )

        # Display each main chunk with its context
        for _, chunk in doc_chunks.iterrows():
            chunk_id = chunk["chunk_id"]

            header = f"\t\tChunk {chunk_id}, Page {chunk['page_number']}, Sequence {chunk['sequence_number']}"
            if has_enhancement_info:
                header += f", {chunk['enhancement_type']}"

            output.append(header)
            output.append(f"\t\tScore: {chunk['score']:.4f}")

            # Get this chunk and its context
            chunk_with_context = doc_chunks[
                (doc_chunks["result_number"] == chunk["result_number"])
            ]

            # Sort by sequence number to get proper order
            chunk_with_context = chunk_with_context.sort_values("sequence_number")

            # Combine all text
            combined_text = "\n".join(chunk_with_context["text"].values)

            # Format text to maximum 80 columns with proper indentation
            wrapped_text = textwrap.fill(
                combined_text, width=80, replace_whitespace=False
            )
            MAX_TEXT_LENGTH = 80 * 10
            if len(wrapped_text) > MAX_TEXT_LENGTH:
                wrapped_text = wrapped_text[:MAX_TEXT_LENGTH] + "\n..."
            output.append(f"\t\tText:")
            output.append(textwrap.indent(wrapped_text, "\t\t"))
            output.append("\n")

    return "\n".join(output)


def filter_results_with_llm(
    results: pd.DataFrame,
    query: str,
    llm_model: str = "gpt-4o-mini",  # gpt-4o-mini is cheaper and faster
    temperature: float = 0,
    task_background: str = "שאלות לגבי תקני הבניה במדינת ישראל לפי מכון התקנים הישראלי עם העדפה לתשובות עובדתיות ומדוייקות",
    n_jobs: int = 10,
) -> pd.DataFrame:
    """
    Filter search results using an LLM to keep only chunks that provide direct information relevant to the query.

    Args:
        results: DataFrame with search results
        query: The original search query
        llm_model: The LLM model to use for filtering
        temperature: Temperature setting for the LLM
        task_background: Background information about the domain to help the LLM understand context
        n_jobs: Number of parallel jobs to run

    Returns:
        DataFrame with filtered results containing only directly relevant chunks
    """
    if results.empty:
        return results

    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    filter_prompt_template = """
בתור מומחה סינון מידע, תפקידך לזהות אם קטע טקסט הבא מספר מידע ישיר וממוקד לשאלה של משתמש.

שאלת המשתמש: {query}


רקע כללי על התחום: {task_background}
עליך לענות אך ורק
TRUE
או
FALSE

קטע הטקסט:
{text}
"""

    def _filter_row(row):
        text = row["text"]
        formatted_prompt = filter_prompt_template.format(
            task_background=task_background, query=query, text=text
        )
        response = llm.invoke(formatted_prompt).text().lower()
        return (row, response)

    n_results_to_filter = len(results)
    n_jobs = min(n_jobs, n_results_to_filter)
    n_jobs = max(n_jobs, 1)
    logger.debug(f"Filtering {n_results_to_filter:,} results with {n_jobs} jobs")
    rows = list(results.reset_index(drop=True).itertuples(index=False, name=None))
    # Convert namedtuples to dicts for compatibility
    rows_dicts = [dict(zip(results.columns, r)) for r in rows]
    # Parallel filtering
    out = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_filter_row)(row) for row in rows_dicts
    )
    filtered_results = []
    lix_kept = []
    for ix, (row, response) in enumerate(out):
        if "true" in response:
            filtered_results.append(row)
            lix_kept.append(ix)

    filtered_df = pd.DataFrame(filtered_results)

    if lix_kept:
        msg = f"LLM filtering: kept {len(filtered_df):,d}/{len(results):,d} out of {n_results_to_filter:,} results"
        for ix in lix_kept[:5]:
            row = results.iloc[ix]
            msg += f" | {ix} ({row.enhancement_type})"
        if len(lix_kept) > 5:
            msg += " ... "
    else:
        msg = f"LLM filtered out all of the initial {n_results_to_filter:,} results"
    logger.info(msg)

    return filtered_df


# @cachier(stale_after=datetime.timedelta(days=1))
def retrieve_documents(
    *,
    query: str = "מה אורך חניית הנכים?",
    top_k: int = 3,
    model_name: Optional[str] = None,
    strategy: Literal[
        "distance_cosine", "distance_euclidean", "similarity_cosine", "similarity_dot"
    ] = "similarity_cosine",
    context: int = 1,
    enhancement_techniques: List[str] = [
        "original",
        "rewrite",
        "decompose",
        "hypothetical_document",
    ],
    task_background: str = "שאלות לגבי תקני הבניה במדינת ישראל לפי מכון התקנים הישראלי עם העדפה לתשובות עובדתיות ומדוייקות",
    filter_with_llm: bool = True,
    llm_filter_model: str = "gpt-4o-mini",
    n_jobs: int = 10,
) -> pd.DataFrame:
    """
    Retrieves relevant documents using query enhancement techniques and vector search.
    All queries and results are expected to be in Hebrew.

    Args:
        query: Original text query to search for (must be Hebrew)
        top_k: Number of results to return per query
        model_name: Name of the embedding model to use
        strategy: Similarity strategy to use
        context: Number of context chunks to retrieve before and after each match
        enhancement_techniques: List of query enhancement techniques to use
            ("original", "rewrite", "decompose", "hypothetical_document")
        task_background: Background information to help guide query enhancement
        filter_with_llm: Whether to use an LLM to filter out results that don't provide
            direct information to the query
        llm_filter_model: The LLM model to use for filtering results
        n_jobs: Number of parallel jobs to run for enhanced queries (default 10, capped at len(enhanced_queries))

    Returns:
        DataFrame with combined search results from all enhanced queries
    """
    # Ensure query is Hebrew (basic check)
    if not any("\u0590" <= c <= "\u05ff" for c in query):
        logger.warning("Query does not appear to be in Hebrew: %s", query)

    logger.info(f"Query: {query}")
    logger.info(f"Task background: {task_background}")
    logger.info(f"Enhancement techniques: {enhancement_techniques}")
    enhancer = QueryEnhancer()
    enhanced_queries = enhancer.create_retrieval_queries(
        query, task_background, techniques=enhancement_techniques
    )
    logger.debug(
        f"Enhanced original query '{query}' into {len(enhanced_queries)} queries"
    )
    for i, q in enumerate(enhanced_queries):
        logger.debug(f"  Query {i+1}: {q}")
    all_results = []
    n_jobs = min(n_jobs, len(enhanced_queries))

    def process_enhanced_query(i, tup_enhanced_query):
        technique, enhanced_query = tup_enhanced_query
        logger.info(
            f"Searching with {technique} query {i+1}/{len(enhanced_queries)}: "
            f"'{hu.get_hebrew_display(enhanced_query)}'"
        )
        EXPANSION_FACTOR = 3
        results = vector_search(
            query=enhanced_query,
            top_k=int(top_k * EXPANSION_FACTOR),
            strategy=strategy,
            context=context,
        )
        logger.info(f"Search with {technique} returned {len(results):,d} results")
        if not results.empty:
            results["source_query"] = enhanced_query
            results["enhancement_type"] = technique
            return results

        return None

    parallel_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_enhanced_query)(i, tup_enhanced_query)
        for i, tup_enhanced_query in enumerate(enhanced_queries)
    )
    all_results = [r for r in parallel_results if r is not None]
    if not all_results:
        logger.info(f"No fragments found for any enhanced query.")
        return pd.DataFrame()
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = (
        combined_results.sort_values("total_score", ascending=False)
        .drop_duplicates(
            subset=["chunk_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    logger.info(
        f"Combined fragments: {combined_results[['chunk_id','score','total_score','title','page_number','sequence_number']].to_dict('records')}"
    )

    if filter_with_llm:
        logger.info(f"Filtering {len(combined_results):,d} results using LLM")
        combined_results = filter_results_with_llm(
            combined_results,
            query,
            llm_model=llm_filter_model,
            task_background=task_background,
        )
        logger.info(f"LLM returned {len(combined_results):,d} results")

    return combined_results


def interactive_document_search(
    *,
    query: str = "כיצד נקבע מספר הרכיבים לבדיקה הראשונה בבניין מגורים?",
    top_k: int = 10,
    model_name: Optional[str] = None,
    strategy: Literal[
        "distance_cosine", "distance_euclidean", "similarity_cosine", "similarity_dot"
    ] = "similarity_cosine",
    context: int = 1,
    enhancement_techniques: List[str] = [
        "original",
        "rewrite",
        "decompose",
        "hypothetical_document",
    ],
    task_background: str = "שאלות לגבי תקני הבניה במדינת ישראל לפי מכון התקנים הישראלי",
    interactive: bool = True,
    filter_with_llm: bool = True,
    llm_filter_model: str = "gpt-4o-mini",
) -> None:
    """
    Search and display documents in an interactive or single-query mode.

    This function retrieves relevant documents based on the query and presents
    them in a user-friendly format. In interactive mode, it also allows the user
    to enter subsequent queries.

    Args:
        query: Text query to search for
        top_k: Number of results to return per enhanced query
        model_name: Name of the embedding model to use
        strategy: Similarity strategy to use
        context: Number of context chunks to retrieve before and after each match
        enhancement_techniques: List of query enhancement techniques to use
            ("original", "rewrite", "decompose", "hypothetical_document")
        task_background: Background information to help guide query enhancement
        interactive: Whether to run in interactive mode. If not interactive, the function will return the results, as well as print them
        filter_with_llm: Whether to use an LLM to filter out results that don't provide direct information to the query
        llm_filter_model: The LLM model to use for filtering results
    """
    # Search with enhanced queries
    results = retrieve_documents(
        query=query,
        top_k=top_k,
        model_name=model_name,
        strategy=strategy,
        context=context,
        enhancement_techniques=enhancement_techniques,
        task_background=task_background,
        filter_with_llm=filter_with_llm,
        llm_filter_model=llm_filter_model,
    )
    formatted_results = format_search_results(
        results, query, strategy, top_k, enhancement_techniques
    )

    if not interactive:
        return formatted_results

    if isinstance(results, pd.DataFrame) and not results.empty:
        formatted_results = format_search_results(
            results, query, strategy, top_k, enhancement_techniques
        )
    else:
        formatted_results = ""
    print(formatted_results)

    # Enter interactive search loop
    while interactive:
        print("\nEnter a search query (or 'q' to quit):")
        user_input = input("> ")
        if user_input.lower() in ("q", "quit", "exit"):
            break

        if user_input.strip():
            results = retrieve_documents(
                query=user_input,
                top_k=top_k,
                model_name=model_name,
                strategy=strategy,
                context=context,
                enhancement_techniques=enhancement_techniques,
                task_background=task_background,
                filter_with_llm=filter_with_llm,
                llm_filter_model=llm_filter_model,
            )

            if isinstance(results, pd.DataFrame) and not results.empty:
                formatted_results = format_search_results(
                    results, user_input, strategy, top_k, enhancement_techniques
                )
            else:
                formatted_results = ""
            print(formatted_results)


# @cachier(stale_after=datetime.timedelta(days=1))
def get_answer(
    query: str,
    top_k: int = 5,
    model_name: Optional[str] = None,
    strategy: Literal[
        "distance_cosine", "distance_euclidean", "similarity_cosine", "similarity_dot"
    ] = "similarity_cosine",
    context: int = 1,
    enhancement_techniques: List[str] = [
        "original",
        "rewrite",
        "decompose",
        "hypothetical_document",
    ],
    task_background: str = "שאלות לגבי תקני הבניה במדינת ישראל לפי מכון התקנים הישראלי עם העדפה לתשובות עובדתיות ומדוייקות",
    llm_model: str = "gpt-4o",
    temperature: float = 0,
    return_source_documents: bool = False,
    filter_with_llm: bool = True,
    llm_filter_model: str = "gpt-4o-mini",
) -> tuple[str, str, pd.DataFrame]:
    """
    Retrieve relevant documents and generate a comprehensive answer to the query.

    Returns:
        Tuple of (answer, sources, fragments):
            answer (str): The generated answer from the LLM.
            sources (str): The formatted context string used for the answer (from format_search_results or web context).
            fragments (pd.DataFrame): The DataFrame of fragments used for the answer (empty if web search or nothing found).
    """
    # Step 1: Retrieve relevant documents
    results_df = retrieve_documents(
        query=query,
        top_k=top_k,
        model_name=model_name,
        strategy=strategy,
        context=context,
        enhancement_techniques=enhancement_techniques,
        task_background=task_background,
        filter_with_llm=filter_with_llm,
        llm_filter_model=llm_filter_model,
    )

    # If we have relevant fragments, use the document-based template
    if not results_df.empty:
        context = format_search_results(
            results_df, query, strategy, top_k, enhancement_techniques
        )
        context_lines = [
            l.strip()
            for l in context.split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
        context = "\n".join(context_lines)
        prompt_template = """
אתה סוכן סיכום מדויק נאמן לאמת.
אני אספק לך בשלב ראשון רקע כללי: {task_background}, ולאחר מכן שאלה: {original_query}.
משימתך היא לספק תשובה מקיפה, מדויקת ועובדתית בהתבסס אך ורק על המידע שבמסמכים שנמסרו.

• אם המידע לא מספיק למענה מלא, ציין זאת בפירוש אך ספק כמה שיותר פרטים רלוונטיים.
• אין להמציא מידע שאינו מופיע במסמכים.
• השפה של התשובה תתאים לשפת השאלה (עברית או אחרת).
• בציטוט השתמש בשם המסמך, במספר העמוד ובמספר הרצף כפי שסופקו.

מבנה מוצע לתשובה:
1. פסקת מבוא קצרה שמסכמת את השאלה.
2. גוף מסודר, מחולק לנקודות או פסקאות לפי נושאים.
3. פסקת סיכום קצרה (אופציונלי).

דוגמה (בעברית):
"על פי המסמך \"תקני בניה\" בעמוד 10, מספר רצף 2, הגובה המרבי המותר לבניין הוא 10 מטר.
לעומת זאת, במסמך \"תקני בניה\" בעמוד 15, מספר רצף 1, הגובה המרבי המותר הוא 8 מטר.
בנוסף, במסמך \"תקני בניה\" בעמוד 12, מספר רצף 3, מצוין שהגובה המותר הוא 12 מטר."

כעת, הנה המסמכים הרלוונטיים:
{context}
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_background", "original_query", "context"],
        )
        formatted_prompt = prompt.format(
            task_background=task_background, original_query=query, context=context
        )
        logger.info(f"Sending prompt to LLM (document-based)")
        llm = ChatOpenAI(model=llm_model, temperature=temperature)
        response = llm.invoke(formatted_prompt)
        log_llm_cost(response, llm_model)
        answer = response.content
        return answer, context, results_df

    # Step 3: Web search fallback
    logger.info("No relevant fragments found, falling back to web search.")
    web_results = web_search_tavily(query)
    if web_results:
        # Format web results as context
        context_lines = []
        for res in web_results:
            line = f"- {res['title']} ({res['url']}, מקור: {res.get('source','')})\n{res['content']}"
            context_lines.append(line)
        context = "\n".join(context_lines)
        prompt_template = """
אתה סוכן סיכום מדויק נאמן לאמת.
אני אספק לך בשלב ראשון רקע כללי: {task_background}, ולאחר מכן שאלה: {original_query}.
משימתך היא לספק תשובה מקיפה, מדויקת ועובדתית בהתבסס אך ורק על המידע שנמצא במקורות האינטרנט שצוינו.

• אם המידע לא מספיק למענה מלא, ציין זאת בפירוש אך ספק כמה שיותר פרטים רלוונטיים.
• אין להמציא מידע שאינו מופיע במקורות.
• השפה של התשובה תתאים לשפת השאלה (עברית או אחרת).
• יש לציין במפורש את שם האתר, כתובת ה-URL, ותאריך הגישה לכל מקור מידע מצוטט.

מבנה מוצע לתשובה:
1. פסקת מבוא קצרה שמסכמת את השאלה.
2. גוף מסודר, מחולק לנקודות או פסקאות לפי נושאים, עם הפניות ברורות למקורות.
3. פסקת סיכום קצרה (אופציונלי).

דוגמה (בעברית):
"על פי האתר 'משרד הבריאות' (https://www.health.gov.il, נגיש בתאריך 1.6.2024), מומלץ לשתות לפחות שני ליטר מים ביום.\nבנוסף, באתר 'כללית' (https://www.clalit.co.il, נגיש בתאריך 1.6.2024) מצוין כי שתייה מספקת תורמת לבריאות הכללית."

כעת, הנה מקורות המידע הרלוונטיים:
{context}
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_background", "original_query", "context"],
        )
        formatted_prompt = prompt.format(
            task_background=task_background, original_query=query, context=context
        )
        logger.info(f"Sending prompt to LLM (web-based)")
        llm = ChatOpenAI(model=llm_model, temperature=temperature)
        response = llm.invoke(formatted_prompt)
        log_llm_cost(response, llm_model)
        answer = response.content
        return answer, context, results_df
    # No web results
    return (
        "לא נמצאו מסמכים רלוונטיים לשאלתך, וגם לא נמצאו תוצאות רלוונטיות בחיפוש אינטרנט.",
        "",
        results_df,
    )


def web_search_tavily(query: str, max_results: int = 5) -> list:
    """
    Perform a web search using the Tavily API for a Hebrew query.
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not set in environment.")
        return []
    client = TavilyClient(api_key)
    response = client.search(query, num_results=max_results, include_images=False)
    # Log Tavily API cost see https://docs.tavily.com/documentation/api-credits
    tavily_cost_usd = 0.00  # currently on the free tier, 1,000 searches per month
    logger.info(f"Tavily API | query={query} | cost_usd={tavily_cost_usd:.6f}")
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
            }
        )
    return results


if __name__ == "__main__":
    # Configure defopt to provide short options for command-line arguments
    defopt.run(
        interactive_document_search,
        short={
            "query": "q",
            "top_k": "k",
            "model_name": "m",
            "context": "c",
            "enhancement_techniques": "e",
            "task_background": "g",
            "interactive": "i",
            "strategy": "s",
        },
    )

    # TODO: We use 4o. 4o mini is cheaper and has the same context window.
    # 4.1 has a much larger context window (1M tokens), is cheaper than 4o
    # 4.1 mini has the same context window as 4o mini, is cheaper than 4o mini
    """
| Model                 | Max context (tokens)   | Input $/1 M tokens   | Output $/1 M tokens  | Modalities            | When it shines                                                                                                              |
|-----------------------|------------------------|----------------------|----------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| GPT-4.1               | 1 000 000              | $2.00                | $8.00                | text                  | Huge corpora or multi-file searches needing > 128 K context                                                                 |
| GPT-4.1 mini          | 1 000 000              | $0.40                | $1.60                | text                  | Same giant contexts at a fraction of the price; slightly lower quality/latency than full 4.1                                |
| GPT-4o                | 128 000               | $5.00                | $20.00               | text + vision + audio | High-quality answers, real-time latency; good for mixed-media pipelines                                                     |
| GPT-4o mini           | 128 000               | $0.60                | $2.40                | text + vision         | Most cost-efficient for mid-size contexts; still beats GPT-3.5 on quality                                                   |
| GPT-3.5-turbo-0125    | 16 000                | $0.50                | $1.50                | text                  | Ultra-cheap for short prompts (< 15 K tokens); good fallback when budget is king                                            |

    """
