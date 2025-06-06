# %%
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from typing import List, Optional, Dict, Any, Tuple

import os
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
assert os.getenv("OPENAI_API_KEY") is not None

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

# Remove import from search.py to avoid circular import
from cost_logging import log_llm_cost, OPENAI_LLM_PRICING

# %%


class SubqueryOutputParser(BaseOutputParser[List[str]]):
    """Parser for extracting sub-queries from LLM output."""

    def parse(self, text: str) -> List[str]:
        """Parse the text into a list of sub-queries."""
        sub_queries = [
            q.strip()
            for q in text.split("\n")
            if q.strip()
            and not q.strip().startswith("Sub-queries:")
            and not q.strip().startswith("example:")
        ]
        return sub_queries


class HypotheticalDocumentOutputParser(BaseOutputParser[List[str]]):
    """Parser for extracting hypothetical documents from LLM output."""

    def parse(self, text: str) -> List[str]:
        """Parse the text into a list of hypothetical documents."""
        # Extract lines that start with a dash or bullet point
        documents = []

        # First try to find lines starting with dash or bullet
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•")):
                # Remove the leading dash or bullet and any whitespace
                documents.append(line[1:].strip())

        # If no bullets found, try to split by newlines assuming each paragraph is a document
        if not documents:
            documents = [
                p.strip()
                for p in text.split("\n\n")
                if p.strip()
                and not p.strip().lower().startswith("hypothetical document")
                and not p.strip().lower().startswith("example:")
            ]

        # If still no documents, treat the entire text as one document
        if not documents:
            documents = [text.strip()]

        return documents


class QueryEnhancer:
    """
    A class that provides various query enhancement techniques for RAG systems.

    This class encapsulates different methods for transforming user queries to improve
    retrieval results in RAG (Retrieval Augmented Generation) systems.

    It can be used directly or as a component in a LangChain pipeline.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 4000,
        custom_llm: Optional[Any] = None,
    ):
        """
        Initialize QueryEnhancer with LLM configuration.

        Args:
            model_name: The OpenAI model to use for query transformations
            temperature: Temperature setting for the model
            max_tokens: Maximum tokens to generate in responses
            custom_llm: Optional custom LLM to use instead of creating a new one
        """
        self.llm = (
            custom_llm
            if custom_llm
            else ChatOpenAI(
                temperature=temperature, model_name=model_name, max_tokens=max_tokens
            )
        )

        # Setup parsers - initialize before using in chain setup
        self.subquery_parser = SubqueryOutputParser()
        self.hypothetical_document_parser = HypotheticalDocumentOutputParser()

        # Initialize all prompts and chains
        self._setup_rewrite_chain()
        self._setup_step_back_chain()
        self._setup_decomposition_chain()
        self._setup_hypothetical_document_chain()

        # Initialize the integrated chains
        self._setup_integrated_chains()

    def _setup_rewrite_chain(self):
        """Set up the query rewriting chain."""
        template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

The rewritten query should be in the same language as the original query.

Keep in mind the general background of the task: {task_background}

Original query: {original_query}

Rewritten query:"""

        self.rewrite_prompt = PromptTemplate(
            input_variables=["original_query", "task_background"], template=template
        )
        self.rewrite_chain = self.rewrite_prompt | self.llm

    def _setup_step_back_chain(self):
        """Set up the step-back prompting chain."""
        template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

The rewritten query should be in the same language as the original query.

Keep in mind the general background of the task: {task_background}

Original query: {original_query}

Step-back query:"""

        self.step_back_prompt = PromptTemplate(
            input_variables=["original_query", "task_background"], template=template
        )
        self.step_back_chain = self.step_back_prompt | self.llm

    def _setup_decomposition_chain(self):
        """Set up the query decomposition chain."""
        template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

The rewritten query should be in the same language as the original query.
Keep in mind the general background of the task: {task_background}

Original query: {original_query}

Don't number the sub-queries.

Example: What are the impacts of climate change on the environment?

Sub-queries:
What are the impacts of climate change on biodiversity?
How does climate change affect the oceans?
What are the effects of climate change on agriculture?
What are the impacts of climate change on human health?"""

        self.decomposition_prompt = PromptTemplate(
            input_variables=["original_query", "task_background"], template=template
        )
        self.decomposition_chain = self.decomposition_prompt | self.llm

        # Create a chain that includes parsing
        self.decomposition_chain_with_parsing = (
            self.decomposition_chain | self.subquery_parser
        )

    def _setup_hypothetical_document_chain(self):
        """Set up the hypothetical document generation chain (HyDE approach)."""
        template = """
אתה עוזר בינה מלאכותית שמטרתו ליצור מסמכים היפותטיים עבור שאילתה נתונה.
המסמכים לא יוצגו למשתמש אלא ישמשו לשיפור שלב האחזור (RAG) על-ידי הדמיה
של "תוצאות מושלמות" לשאילתה.

הנחיות-יסוד:
1. המסמכים חייבים להיות מפורטים, מדויקים ורלוונטיים ישירות לשאילתה.
2. כתוב אותם באותה שפה של השאילתה.
3. כל ערך מספרי (שיפועים, מידות, טווחים) חייב לכלול יחידות מדויקות
   -- למשל 2%, 1.60 מ', 160 ס"מ.
4. היה חד-משמעי לגבי דרישות מינימום/מקסימום.
5. הפק מסמכים בני שורה אחת (בולטים), 3-5 במספר, המתחילים במקף (-).

רקע כללי: {task_background}

שאילתה מקורית: {original_query}

צור 3-5 מסמכים היפותטיים שיענו בצורה מושלמת על השאילתה. השתמש בידע הכללי כדי ליצור מסמכים מדויקים ומפורטים.

---

דוגמה:
שאילתה: "מהן דרישות המינימום לרוחב דלת נגישה?"

מסמכים היפותטיים:
- רוחב מעבר חופשי מינימלי בדלת נגישה הוא 90 ס"מ.
- כאשר הדלת משמשת יציאת חירום, התקן מחמיר ל-100 ס"מ רוחב מעבר חופשי.
- דלת דו-כנפית נחשבת נגישה אם אחת הכנפיים מספקת רוחב מעבר של 90 ס"מ לפחות.
- בדלתות הזזה דרוש מסלול תנועה חופשי של 95 ס"מ כדי למנוע חסימה בחריץ ההזזה.
- ידיות בדלת נגישה חייבות להיות במרחק 50-110 ס"מ מהרצפה ואסור שיפחיתו מרוחב המעבר שנקבע.
"""

        self.hypothetical_document_prompt = PromptTemplate(
            input_variables=["original_query", "task_background"], template=template
        )

        # Create a chain that includes parsing to get a list of hypothetical documents
        self.hypothetical_document_chain = (
            self.hypothetical_document_prompt
            | self.llm
            | self.hypothetical_document_parser
        )

    def _setup_integrated_chains(self):
        """Set up integrated chains that can be used in LangChain pipelines."""

        # Create a parallel chain that runs all enhancements at once
        self.all_enhancements_chain = RunnableParallel(
            {
                "original": lambda x: x["original_query"],
                "rewritten": self.rewrite_chain,
                "step_back": self.step_back_chain,
                "sub_queries": self.decomposition_chain_with_parsing,
                "hypothetical_document": self.hypothetical_document_chain,
            }
        )

    def rewrite_query(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
    ) -> str:
        """
        Rewrite the original query to be more specific and detailed.

        Args:
            original_query: The original user query
            task_background: The background information for the query

        Returns:
            A rewritten, more specific query
        """
        response = self.rewrite_chain.invoke(
            {"original_query": original_query, "task_background": task_background}
        )  # response is the LLM object, not a string
        log_llm_cost(response, getattr(self.llm, "model_name", "gpt-4o"))
        return response.content

    def generate_step_back_query(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
    ) -> str:
        """
        Generate a more general, step-back query to retrieve broader context.

        Args:
            original_query: The original user query
            task_background: The background information for the query

        Returns:
            A more general step-back query
        """
        response = self.step_back_chain.invoke(
            {"original_query": original_query, "task_background": task_background}
        )
        log_llm_cost(response, getattr(self.llm, "model_name", "gpt-4o"))
        return response.content

    def decompose_query(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
    ) -> List[str]:
        """
        Decompose the original query into simpler sub-queries.
        """
        # Get raw LLM response first
        raw_response = self.decomposition_chain.invoke(
            {"original_query": original_query, "task_background": task_background}
        )
        log_llm_cost(raw_response, getattr(self.llm, "model_name", "gpt-4o"))
        # Now parse
        parsed = self.subquery_parser.parse(raw_response.content)
        return parsed

    def generate_hypothetical_document(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
    ) -> List[str]:
        """
        Generate hypothetical documents that would be perfect answers to the query (HyDE approach).

        This uses the Hypothetical Document Embeddings (HyDE) approach described in
        "Precise Zero-Shot Dense Retrieval without Relevance Labels" by Gao et al.

        The generated documents can be embedded and used for retrieval instead of
        the original query, potentially improving retrieval performance.

        Args:
            original_query: The original user query
            task_background: The background information for the query

        Returns:
            A list of hypothetical documents that answer the query
        """
        # Get raw LLM response first
        raw_response = self.hypothetical_document_prompt | self.llm
        llm_response = raw_response.invoke(
            {"original_query": original_query, "task_background": task_background}
        )
        log_llm_cost(llm_response, getattr(self.llm, "model_name", "gpt-4o"))
        # Now parse
        parsed = self.hypothetical_document_parser.parse(llm_response.content)
        return parsed

    def enhance_query(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
        techniques: List[str] = [
            "rewrite",
            "step_back",
            "decompose",
            "hypothetical_document",
        ],
    ) -> Dict[str, Any]:
        """
        Apply multiple query enhancement techniques at once.

        Args:
            original_query: The original user query
            task_background: The background information for the query
            techniques: List of techniques to apply ("rewrite", "step_back", "decompose", "hypothetical_document")

        Returns:
            A dictionary containing the results of all requested enhancements
        """
        results = {"original": original_query}
        logger.info(
            f"Enhancing query: {original_query} | Task background: {task_background} | Techniques: {techniques}"
        )
        if "rewrite" in techniques:
            results["rewritten"] = self.rewrite_query(original_query, task_background)
            logger.info(f"Rewritten: {results['rewritten']}")
        if "step_back" in techniques:
            results["step_back"] = self.generate_step_back_query(
                original_query, task_background
            )
            logger.info(f"Step-back: {results['step_back']}")
        if "decompose" in techniques:
            results["sub_queries"] = self.decompose_query(
                original_query, task_background
            )
            logger.info(f"Sub-queries: {results['sub_queries']}")
        if "hypothetical_document" in techniques:
            results["hypothetical_document"] = self.generate_hypothetical_document(
                original_query, task_background
            )
            logger.info(f"Hypothetical documents: {results['hypothetical_document']}")
        return results

    def get_langchain_runnable(
        self, techniques: List[str] = ["rewrite", "step_back", "decompose"]
    ) -> Any:
        """
        Get a LangChain Runnable that can be integrated into a pipeline.

        Args:
            techniques: List of techniques to include in the chain

        Returns:
            A LangChain Runnable that can be used in a pipeline
        """
        components = {"original": lambda x: x["original_query"]}

        if "rewrite" in techniques:
            components["rewritten"] = self.rewrite_chain

        if "step_back" in techniques:
            components["step_back"] = self.step_back_chain

        if "decompose" in techniques:
            components["sub_queries"] = self.decomposition_chain_with_parsing

        if "hypothetical_document" in techniques:
            components["hypothetical_document"] = self.hypothetical_document_chain

        return RunnableParallel(components)

    def create_retrieval_queries(
        self,
        original_query: str,
        task_background: str = "General information retrieval",
        techniques: List[str] = ["original", "rewrite", "step_back"],
        flatten_subqueries: bool = True,
        flatten_hypothetical_documents: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Create a list of all queries to use for retrieval.

        Args:
            original_query: The original user query
            task_background: The background information for the query
            techniques: List of techniques to include ("original", "rewrite", "step_back", "decompose", "hypothetical_document")
            flatten_subqueries: Whether to include sub-queries as individual queries in the result
            flatten_hypothetical_documents: Whether to include each hypothetical document as an individual query

        Returns:
            A list of tuples, each containing the technique name and the query
        """
        queries = []
        results = self.enhance_query(
            original_query,
            task_background,
            techniques=[t for t in techniques if t != "original"],
        )

        if "original" in techniques:
            queries.append(("original", original_query))

        if "rewrite" in techniques and "rewritten" in results:
            queries.append(("rewrite", results["rewritten"]))

        if "step_back" in techniques and "step_back" in results:
            queries.append(("step_back", results["step_back"]))

        if "hypothetical_document" in techniques and "hypothetical_document" in results:
            if flatten_hypothetical_documents and isinstance(
                results["hypothetical_document"], list
            ):
                # Add each hypothetical document as a separate query
                queries.extend(
                    [
                        ("hypothetical_document", doc)
                        for doc in results["hypothetical_document"]
                    ]
                )
            else:
                # If not flattening or if it's not a list, add the whole result
                doc_content = results["hypothetical_document"]
                if isinstance(doc_content, list):
                    # Join the list into a single string if needed
                    doc_content = "\n".join(doc_content)
                queries.append(("hypothetical_document", doc_content))

        if (
            "decompose" in techniques
            and "sub_queries" in results
            and flatten_subqueries
        ):
            queries.extend([("decompose", q) for q in results["sub_queries"]])

        return queries


# Example of integrating with a LangChain RAG pipeline
def rag_pipeline_example():
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # Setup QueryEnhancer
    enhancer = QueryEnhancer()

    # Example: Create a retriever that uses both original and rewritten queries
    # This is simplified - you'd typically have a proper vectorstore setup
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Define a function to get enhanced queries and retrieve documents for each
    def retrieve_with_enhanced_queries(input_dict):
        queries = enhancer.create_retrieval_queries(
            input_dict["query"],
            input_dict["task_background"],
            techniques=["original", "rewrite", "step_back"],
        )

        # Retrieve documents for each query and deduplicate
        all_docs = []
        seen_ids = set()

        for query in queries:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.metadata.get("id") not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc.metadata.get("id"))

        return {"docs": all_docs, "query": input_dict["query"]}

    # Define a simple RAG pipeline with query enhancement
    rag_chain = (
        RunnablePassthrough()
        | retrieve_with_enhanced_queries
        | (lambda x: f"Context: {x['docs']}\n\nQuestion: {x['query']}")
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    # This is how you would use the pipeline
    # result = rag_chain.invoke({"query": "What is climate change?", "task_background": "Environmental science questions"})


# Example usage
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.WARN,
    )

    task_background = (
        "שאלות לגבי תקני הבניה במדינת ישראל עם העדפה לתשובות עובדתיות ומדויקות"
    )

    original_query = "ממה שיפוע נדרש במרפסות ובמקלחונים בדירה?"

    # Create an instance of QueryEnhancer
    enhancer = QueryEnhancer()

    # Get all query enhancements at once
    results = enhancer.enhance_query(
        original_query,
        task_background,
        techniques=[
            "rewrite",
            "step_back",
            "decompose",
            "hypothetical_document",
        ],
    )

    # Print the results
    logging.info(f"Original:\n{results['original']}\n")
    logging.info(f"Rewritten:\n{results['rewritten']}\n")
    logging.info(f"Step-back:\n{results['step_back']}\n")
    logging.info(f"Hypothetical Documents:")
    for i, doc in enumerate(results["hypothetical_document"], 1):
        logging.info(f"{i}. {doc}")
    logging.info(f"\nSub-queries:")
    for i, q in enumerate(results["sub_queries"], 1):
        logging.info(f"{i}. {q}")

    # Example of getting a list of all queries for retrieval
    logging.info("\nAll retrieval queries:")
    all_queries = enhancer.create_retrieval_queries(
        original_query,
        task_background,
        techniques=[
            "original",
            "rewrite",
            "step_back",
            "decompose",
            "hypothetical_document",
        ],
    )
    for i, q in enumerate(all_queries, 1):
        logging.info(f"{i}. {q[0]}: {q[1]}")

# %%
