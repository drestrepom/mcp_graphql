from enum import Enum
from typing import (
    ForwardRef,
    Optional,
    TypeVar,
)

from pydantic import BaseModel, Field

from mcp_graphql.graphql_client import GraphQLClient

# Type variable for generic models
T = TypeVar("T")

# Forward references for circular dependencies
QueryRef = ForwardRef("Query")
AuthorRef = ForwardRef("Author")
QuoteConnectionRef = ForwardRef("QuoteConnection")
QuoteEdgeRef = ForwardRef("QuoteEdge")
QuoteRef = ForwardRef("Quote")
PageInfoRef = ForwardRef("PageInfo")
AuthorConnectionRef = ForwardRef("AuthorConnection")
AuthorEdgeRef = ForwardRef("AuthorEdge")
AuthorsOrderRef = ForwardRef("AuthorsOrder")
MutationRef = ForwardRef("Mutation")
NewAuthorRef = ForwardRef("NewAuthor")
deleteAuthorInputRef = ForwardRef("deleteAuthorInput")


class AuthorsOrderField(str, Enum):
    ID = "ID"
    FIRST_NAME = "FIRST_NAME"
    LAST_NAME = "LAST_NAME"
    CREATED_AT = "CREATED_AT"


"""The ordering direction."""


class Direction(str, Enum):
    ASC = "ASC"  # Specifies an ascending order for a given orderBy argument.
    DESC = "DESC"  # Specifies a descending order for a given orderBy argument.


class AuthorsOrder(BaseModel):
    field: "AuthorsOrderField" = Field(...)
    direction: "Direction" = Field(...)


class NewAuthor(BaseModel):
    first_name: str = Field(..., alias="firstName")
    last_name: str = Field(..., alias="lastName")

    class Config:
        populate_by_name = True


class deleteAuthorInput(BaseModel):
    id: int = Field(...)


class authorArgs(BaseModel):
    id: str = Field(...)


class authorsArgs(BaseModel):
    first: int | None = Field(
        None,
        description="Limits the number of results returned in the page. Defaults to 10.",
    )
    """Limits the number of results returned in the page. Defaults to 10."""
    after: str | None = Field(
        None,
        description="The cursor value of an item returned in previous page. An alternative to in integer offset.",
    )
    """The cursor value of an item returned in previous page. An alternative to in integer offset."""
    first_name: str | None = Field(None, alias="firstName")
    last_name: str | None = Field(None, alias="lastName")
    order_by: list[Optional["AuthorsOrder"]] | None = Field(None, alias="orderBy")

    class Config:
        populate_by_name = True


class quoteArgs(BaseModel):
    id: str = Field(...)


class quotesArgs(BaseModel):
    first: int | None = Field(
        None,
        description="Limits the number of results returned in the page. Defaults to 10.",
    )
    """Limits the number of results returned in the page. Defaults to 10."""
    after: str | None = Field(
        None,
        description="The cursor value of an item returned in previous page. An alternative to in integer offset.",
    )
    """The cursor value of an item returned in previous page. An alternative to in integer offset."""
    query: str | None = Field(None)


class createAuthorArgs(BaseModel):
    input: "NewAuthor" = Field(...)


class deleteAuthorArgs(BaseModel):
    input: "deleteAuthorInput" = Field(...)


class Query(GraphQLClient, BaseModel):
    # Definir url como un campo de Pydantic para que no cause error al asignarlo
    url: str | None = Field(default=None, exclude=True)

    def __init__(self, url: str | None = None, **data) -> None:
        GraphQLClient.__init__(self, url)
        BaseModel.__init__(self, **data)

    # GraphQL query/mutation methods
    def author(self, args: authorArgs) -> "Author":
        # Determine fields to request based on return type
        fields = self.get_model_fields("Author")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_query("Author", "author", fields, variables)

    def authors(self, args: authorsArgs) -> "AuthorConnection":
        # Determine fields to request based on return type
        fields = self.get_model_fields("AuthorConnection")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_query("AuthorConnection", "authors", fields, variables)

    def quote(self, args: quoteArgs) -> "Quote":
        # Determine fields to request based on return type
        fields = self.get_model_fields("Quote")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_query("Quote", "quote", fields, variables)

    def quotes(self, args: quotesArgs) -> "QuoteConnection":
        # Determine fields to request based on return type
        fields = self.get_model_fields("QuoteConnection")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_query("QuoteConnection", "quotes", fields, variables)


class Author(BaseModel):
    def id(self) -> str:
        """Globally unique ID of the author"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def db_id(self) -> str:
        """Database ID of the author"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def first_name(self) -> str:
        """Author\'s first name"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def last_name(self) -> str:
        """Author\'s last name"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def quotes(self, args: quotesArgs) -> "QuoteConnection":
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def created_at(self) -> str:
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    # GraphQL subscription methods
    # GraphQL subscription methods


class QuoteConnection(BaseModel):
    def total_count(self) -> int:
        """Identifies the total count of items in the connection."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def edges(self) -> list[Optional["QuoteEdge"]]:
        """A list of edges."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def page_info(self) -> "PageInfo":
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    # GraphQL subscription methods


"""List of edges."""


class QuoteEdge(BaseModel):
    def node(self) -> "Quote":
        """The item at the end of the edge."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def cursor(self) -> str:
        """A cursor for pagination."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)


class Quote(BaseModel):
    def id(self) -> str:
        """Globally unique ID of the quote"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def db_id(self) -> str:
        """Database ID of the quote"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def text(self) -> str:
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def author(self) -> "Author":
        """Author of the quote"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def created_at(self) -> str:
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)


"""Information about pagination in a connection."""


class PageInfo(BaseModel):
    def end_cursor(self) -> str:
        """The item at the end of the edge."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def has_next_page(self) -> bool:
        """When paginating forwards, are there more items?"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def has_previous_page(self) -> bool:
        """When paginating backwards, are there more items?"""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def start_cursor(self) -> str:
        """When paginating backwards, the cursor to continue."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    # GraphQL subscription methods


class AuthorConnection(BaseModel):
    def total_count(self) -> int:
        """Identifies the total count of items in the connection."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def edges(self) -> list[Optional["AuthorEdge"]]:
        """A list of edges."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def page_info(self) -> "PageInfo":
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)


"""List of edges."""


class AuthorEdge(BaseModel):
    def node(self) -> "Author":
        """The item at the end of the edge."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    def cursor(self) -> str:
        """A cursor for pagination."""
        msg = "Subscriptions not yet implemented"
        raise NotImplementedError(msg)

    # GraphQL subscription methods
    # GraphQL subscription methods
    # GraphQL subscription methods


class Mutation(GraphQLClient, BaseModel):
    # Definir url como un campo de Pydantic para que no cause error al asignarlo
    url: str | None = Field(default=None, exclude=True)

    def __init__(self, url: str | None = None, **data) -> None:
        GraphQLClient.__init__(self, url)
        BaseModel.__init__(self, **data)

    # GraphQL query/mutation methods
    def create_author(self, args: createAuthorArgs) -> "Author":
        # Determine fields to request based on return type
        fields = self.get_model_fields("Author")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_mutation("Author", "createAuthor", fields, variables)

    def delete_author(self, args: deleteAuthorArgs) -> "Author":
        # Determine fields to request based on return type
        fields = self.get_model_fields("Author")
        # Use model_dump for Pydantic v2
        variables = args.model_dump(by_alias=True) if args else None
        return self.execute_mutation("Author", "deleteAuthor", fields, variables)

    # GraphQL subscription methods
    # GraphQL subscription methods


# Resolve forward references
Query.update_forward_refs()
Author.update_forward_refs()
QuoteConnection.update_forward_refs()
QuoteEdge.update_forward_refs()
Quote.update_forward_refs()
PageInfo.update_forward_refs()
AuthorConnection.update_forward_refs()
AuthorEdge.update_forward_refs()
AuthorsOrder.update_forward_refs()
Mutation.update_forward_refs()
NewAuthor.update_forward_refs()
deleteAuthorInput.update_forward_refs()
