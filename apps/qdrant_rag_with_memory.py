from dataclasses import dataclass, field
from typing_extensions import TypedDict
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_ollama import ChatOllama

from langgraph.graph import START, END, StateGraph
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class ChatbotWithMemory:
    def __init__(self, llm, retriever):
        # Setup
        self.llm = llm
        self.custom_retriever = retriever
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Builds and returns the LangGraph computation graph with retrieve, query_or_respond, and generate steps.
        """
        graph_builder = StateGraph(MessagesState)

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.custom_retriever.invoke(query)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([retrieve])
            response = llm_with_tools.invoke(state["messages"])
            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}
        
        # Step 2: Execute the retrieval.
        tools = ToolNode([retrieve])

        # Step 3: Generate a response using the retrieved content.
        def generate(state: MessagesState):
            """Generate answer."""
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
        
            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved-context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Keep the "
                "answer as detailed as possible and factual. Only stick to the retrieved context."
                "\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages
        
            # Run
            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)
        
        memory = MemorySaver()
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        graph = graph_builder.compile(checkpointer=memory)
        return graph
   
    def get_history(self, config):
         return next(self.graph.get_state_history(config))
        
    def query(self, user_input, config):
        """
        Processes user input and returns AI-generated responses.

        :param user_input: The user's query.
        :return: The latest AI response.
        """
        for step in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            messages = step.get("messages", [])  # Get messages safely
        
            # Get the last message and check if it's an AIMessage
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].content:
                return messages[-1].content.strip()  
                
        return "No response from AI."  # Fallback