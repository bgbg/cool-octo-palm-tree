from typing import List, Optional
from typing_extensions import Literal
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    ForeignKey,
    DateTime,
    Text,
    Boolean,
    Index,
    UniqueConstraint,
    JSON,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.types import UserDefinedType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
import logging


# Define a custom Vector type for pgvector
class VECTOR(UserDefinedType):
    def get_col_spec(self, **kw):
        return "vector"

    def bind_processor(self, dialect):
        def process(value):
            return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            return value

        return process


Base = declarative_base()


class Document(Base):
    """Document table for storing metadata about uploaded documents."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("title", "storage_path", name="uix_title_storage_path"),
    )

    # Relationships
    elements = relationship(
        "Element", back_populates="document", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Element(Base):
    """Element table for storing parsed elements from documents."""

    __tablename__ = "elements"

    element_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    type = Column(Text)  # e.g., NarrativeText, Title, CompositeElement
    text = Column(Text, nullable=False)
    page_number = Column(Integer)

    # Coordinates metadata
    coordinates_layout_height = Column(Integer)
    coordinates_layout_width = Column(Integer)
    coordinates_points = Column(JSONB)
    coordinates_system = Column(Text)

    # Detection metadata
    detection_class_prob = Column(Float)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document = relationship("Document", back_populates="elements")
    chunks = relationship("ElementChunkMapping", back_populates="element")


class Chunk(Base):
    """Chunk table for storing text chunks generated from documents."""

    __tablename__ = "chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    type = Column(Text)  # Always CompositeElement, Table, or TableChunk
    page_number = Column(Integer)
    sequence_number = Column(Integer)  # Running number within document

    # Original elements that make up this chunk
    orig_elements = Column(Text)  # Base64 gzipped JSON of original elements

    # Chunking metadata
    chunk_method_label = Column(Text)  # e.g., max_characters=300, overlap=0

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    document = relationship("Document", back_populates="chunks")
    elements = relationship(
        "ElementChunkMapping", back_populates="chunk", cascade="all, delete-orphan"
    )
    embeddings = relationship(
        "Embedding", back_populates="chunk", cascade="all, delete-orphan"
    )


class ElementChunkMapping(Base):
    """Join table for many-to-many relationship between chunks and elements."""

    __tablename__ = "element_chunk_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.chunk_id"), nullable=False)
    element_id = Column(
        UUID(as_uuid=True), ForeignKey("elements.element_id"), nullable=False
    )
    position = Column(Integer)  # Order of elements within the chunk

    # Relationships
    chunk = relationship("Chunk", back_populates="elements")
    element = relationship("Element", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("chunk_id", "element_id", name="uix_chunk_element"),
    )


class Embedding(Base):
    """Embedding table for storing vector embeddings of chunks."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.chunk_id"), nullable=False)
    embedding = Column(VECTOR)  # Using PostgreSQL VECTOR type
    model_name = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    chunk = relationship("Chunk", back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint(
            "chunk_id", "model_name", "model_version", name="uix_chunk_model"
        ),
    )


def get_db_engine(connection_string: str):
    """Create a SQLAlchemy engine from a connection string.

    Args:
        connection_string: Database connection string

    Returns:
        SQLAlchemy engine
    """
    return create_engine(connection_string)


def get_db_session(engine) -> Session:
    """Create a new SQLAlchemy session.

    Args:
        engine: SQLAlchemy engine

    Returns:
        SQLAlchemy session
    """
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine):
    """Initialize database by creating all tables.

    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)


def add_data(
    session: Session,
    document_data: dict,
    elements_data: List[dict],
    chunks_data: List[dict],
    embeddings_data: Optional[List[dict]] = None,
) -> Document:
    """
    Add a document with its elements, chunks, and optional embeddings to the database.

    Args:
        session: SQLAlchemy session
        document_data: Dictionary containing document data
                      (keys: title, storage_path)
        elements_data: List of dictionaries containing element data
                      (keys: type, text, page_number, etc.)
        chunks_data: List of dictionaries containing chunk data
                    (keys: text, type, page_number, chunking_strategy, etc.)
        embeddings_data: Optional list of dictionaries containing embedding data
                        (keys: embedding, model_name, model_version)

    Returns:
        The created Document instance
    """
    # Create document
    document = Document(
        title=document_data["title"], storage_path=document_data["storage_path"]
    )
    session.add(document)
    session.flush()  # Flush to get the document ID
    logging.debug(f"Created document: {document.id}")

    # Create elements
    elements = []
    for element_data in elements_data:
        element = Element(document_id=document.id, **element_data)
        session.add(element)
        elements.append(element)
    session.flush()  # Flush to get element IDs
    logging.debug(f"Created {len(elements)} elements")

    # Map of element texts to their IDs for element_chunk_mappings association
    element_map = {element.text: element.element_id for element in elements}
    logging.debug(f"Created element map with {len(element_map)} entries")

    # Log some sample element texts for debugging
    sample_texts = list(element_map.keys())[:3]
    for i, text in enumerate(sample_texts):
        truncated = text[:30] + "..." if len(text) > 30 else text
        logging.debug(f"Element map sample {i}: '{truncated}' -> {element_map[text]}")

    # Create chunks first without creating mappings
    chunks = []
    for i, chunk_data in enumerate(chunks_data):
        # Extract element_texts from chunk data if present but keep it for later use
        element_texts = chunk_data.pop("element_texts", [])
        # Store element_texts in a separate attribute to use later
        chunk_data["_element_texts"] = element_texts

        chunk = Chunk(
            document_id=document.id,
            **{k: v for k, v in chunk_data.items() if k != "_element_texts"},
        )
        session.add(chunk)
        chunks.append(chunk)

    # Flush to ensure all chunks have IDs
    session.flush()
    logging.debug(f"Created {len(chunks)} chunks")

    # Now create element_chunk_mappings with the confirmed chunk IDs
    total_mappings = 0
    for i, chunk in enumerate(chunks):
        chunk_data = chunks_data[i]
        element_texts = chunk_data.get("_element_texts", [])

        # Log information about this chunk
        has_texts = len(element_texts) > 0
        logging.debug(f"Chunk {i}: Has {len(element_texts)} element_texts")

        # Create element_chunk_mapping associations
        chunk_mappings = 0
        for position, text in enumerate(element_texts):
            if text in element_map:
                # Ensure both the chunk_id and element_id are valid UUIDs
                if chunk.chunk_id and element_map[text]:
                    mapping = ElementChunkMapping(
                        chunk_id=chunk.chunk_id,
                        element_id=element_map[text],
                        position=position,
                    )
                    session.add(mapping)
                    chunk_mappings += 1
                    total_mappings += 1
                else:
                    logging.warning(
                        f"Invalid IDs: chunk_id={chunk.chunk_id}, element_id={element_map[text]}"
                    )
            else:
                # Log when we can't find an element
                truncated = text[:50] + "..." if len(text) > 50 else text
                logging.warning(f"Cannot find element for text: '{truncated}'")

        if chunk_mappings > 0:
            logging.debug(f"Created {chunk_mappings} mappings for chunk {i}")
        elif len(element_texts) > 0:
            logging.warning(
                f"No mappings created for chunk {i} despite having {len(element_texts)} element_texts"
            )

    session.flush()  # Flush to commit the mappings
    logging.debug(
        f"Created {len(chunks)} chunks and {total_mappings} element-chunk mappings"
    )

    # Create embeddings if provided
    if embeddings_data:
        # Map chunks to their IDs for convenience
        chunk_map = {chunk.text: chunk.chunk_id for chunk in chunks}

        for embedding_data in embeddings_data:
            # Extract chunk_text from embedding data
            chunk_text = embedding_data.pop("chunk_text", None)
            chunk_id = embedding_data.pop("chunk_id", None)

            # Determine chunk_id either directly or via text lookup
            if not chunk_id and chunk_text in chunk_map:
                chunk_id = chunk_map[chunk_text]

            if chunk_id:
                embedding = Embedding(chunk_id=chunk_id, **embedding_data)
                session.add(embedding)

    session.commit()
    return document


def add_embeddings(
    session: Session,
    embeddings_data: dict[uuid.UUID, List[float]],
    model_name: str = "UNDEF",
    model_version: str = "UNDEF",
    if_exists: Literal["update", "skip", "raise"] = "skip",
) -> List[Embedding]:
    """
    Add new embeddings for existing chunks in the database.

    Args:
        session: SQLAlchemy session
        embeddings_data: Dictionary where keys are chunk_ids (UUID) and values are embedding vectors
        model_name: Name of the embedding model, defaults to "UNDEF"
        model_version: Version of the embedding model, defaults to "UNDEF"
        if_exists: Strategy for handling existing embeddings:
                  - "update": Update existing embeddings with new values
                  - "skip": Skip existing embeddings (default)
                  - "raise": Raise an error if any embedding already exists

    Returns:
        List of created or updated Embedding instances

    Raises:
        ValueError: If if_exists="raise" and an embedding already exists
    """
    created_or_updated_embeddings = []

    # If if_exists="raise", check all embeddings first
    if if_exists == "raise":
        for chunk_id in embeddings_data:
            existing = (
                session.query(Embedding)
                .filter(
                    Embedding.chunk_id == chunk_id,
                    Embedding.model_name == model_name,
                    Embedding.model_version == model_version,
                )
                .first()
            )

            if existing:
                session.rollback()
                raise ValueError(
                    f"Embedding already exists for chunk_id {chunk_id} with model {model_name}:{model_version}"
                )

    for chunk_id, embedding_vector in embeddings_data.items():
        # Check if the embedding already exists for this chunk and model
        existing = (
            session.query(Embedding)
            .filter(
                Embedding.chunk_id == chunk_id,
                Embedding.model_name == model_name,
                Embedding.model_version == model_version,
            )
            .first()
        )

        if existing:
            if if_exists == "update":
                # Update the existing embedding
                existing.embedding = embedding_vector
                created_or_updated_embeddings.append(existing)
            # For "skip", do nothing
            # For "raise", we already checked above
        else:
            # Create new embedding
            embedding = Embedding(
                chunk_id=chunk_id,
                embedding=embedding_vector,
                model_name=model_name,
                model_version=model_version,
            )
            session.add(embedding)
            created_or_updated_embeddings.append(embedding)

    session.commit()
    return created_or_updated_embeddings


def remove_document_data(session: Session, document_id: uuid.UUID) -> bool:
    """
    Remove a document and all its associated data (elements, chunks, element_chunk_mappings, embeddings).

    Args:
        session: SQLAlchemy session
        document_id: UUID of the document to remove

    Returns:
        bool: True if the document was found and removed, False if not found

    Raises:
        Exception: If there's an error during deletion
    """
    try:
        # Find the document
        document = session.query(Document).filter(Document.id == document_id).first()

        if not document:
            return False

        # SQLAlchemy handles deletion of related records through cascade
        # We just need to delete the document
        session.delete(document)
        session.commit()
        return True

    except Exception as e:
        session.rollback()
        raise e
