"""
Scribe — Generates formal, structured legal responses using Gemini.

Takes retrieved legal context and conversation history to produce
grounded, well-cited legal answers.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from models.semantic_cache import store as cache_store
from utils.config import get_llm
from controllers.base_agent import BaseAgent


SCRIBE_PROMPT_AR = ChatPromptTemplate.from_messages([
    ("system", """أنت مستشار قانوني متخصص في القانون المدني المصري. عليك تقديم إجابات دقيقة ومنظمة بناءً على النصوص القانونية المتاحة.

## القواعد:
1. استند فقط إلى النصوص القانونية المقدمة في السياق (إلا في حالة إلقاء التحية).
2. اذكر أرقام المواد عند الإشارة إليها.
3. إذا لم تجد إجابة في النصوص المقدمة، قل ذلك بوضوح.
4. استخدم لغة قانونية رسمية.
5. قدم تفسيراً عملياً عند الإمكان.
6. نظم إجابتك بشكل واضح مع عناوين فرعية عند الحاجة.
7. إذا كان المستخدم يلقي التحية فقط (مثل: "مرحبا"، "أهلا"، "السلام عليكم")، قم بالرد بترحيب لطيف، وعرف بنفسك كمستشار قانوني مصري، واسأله كيف يمكنك مساعدته، وتجاهل عدم وجود نصوص قانونية.

## سياق الجودة:
- عدد الوثائق المستردة: {num_docs}
- إذا كان max_relevance < 0.5، أشر إلى أن الإجابة قد تكون غير كاملة أو غير دقيقة.

## السياق القانوني:
{context}

## سجل المحادثة:
{history}
"""),
    ("human", "{query}"),
])


SCRIBE_PROMPT_EN = ChatPromptTemplate.from_messages([
    ("system", """You are a legal advisor specializing in the Egyptian Civil Code. Provide accurate, structured answers based on the available legal texts.

## Rules:
1. Base your answer ONLY on the provided legal context (except for greetings).
2. Cite article numbers when referencing specific provisions.
3. If the answer is not found in the provided context, clearly state so.
4. Use formal legal language.
5. Provide practical interpretation when possible.
6. Structure your answer clearly with subheadings when needed.
7. Since the source texts are in Arabic, you may include the original Arabic text alongside your English explanation for accuracy.
8. If the user is simply greeting you (e.g., "Hello", "Hi"), warmly introduce yourself as an Egyptian Civil Code legal advisor and ask how you can help, ignoring the lack of legal context.

## Quality Context:
- Number of documents retrieved: {num_docs}
- Maximum relevance: {max_relevance}
- If max_relevance < 0.5, indicate that the answer may be incomplete or uncertain.

## Legal Context:
{context}

## Conversation History:
{history}
"""),
    ("human", "{query}"),
])


class ScribeAgent(BaseAgent):
    """
    Agent responsible for generating formal, structured legal responses.
    """
    
    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the prompt."""
        if not documents:
            return "لا توجد نصوص قانونية متاحة. / No legal texts available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            article = meta.get("article_number", "N/A")
            category = meta.get("category", "general")

            header = f"[مصدر/Source: {source} | مادة/Article: {article} | تصنيف/Category: {category}]"
            context_parts.append(f"--- Document {i} ---\n{header}\n{doc.page_content}\n")

        return "\n".join(context_parts)

    def _format_history(self, messages: list) -> str:
        """Format conversation history for the prompt."""
        if not messages:
            return "لا يوجد سجل محادثة سابق. / No prior conversation history."

        history_parts = []
        for msg in messages[-3:]:  # Keep last 3 messages for context window
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                history_parts.append(f"المستخدم/User: {content}")
            elif role == "assistant":
                history_parts.append(f"المستشار/Advisor: {content[:500]}...")  # Truncate long answers

        return "\n".join(history_parts)

    def execute(
        self,
        query: str,
        retrieved_docs: list[Document],
        conversation_history: list = None,
        language: str = "ar",
        cache_result: bool = True,
        num_docs: int = 0,
        max_relevance: float = 1.0,
    ) -> str:
        """
        Generate a formal legal response based on retrieved context.
        
        Args:
            query: The user's legal query.
            retrieved_docs: List of relevant Document objects from retrieval.
            conversation_history: List of past messages [{"role": ..., "content": ...}].
            language: Response language ("ar" or "en").
            cache_result: Whether to store the result in semantic cache.
        
        Returns:
            The generated legal response as a string.
        """
        llm = get_llm()

        context = self._format_context(retrieved_docs)
        history = self._format_history(conversation_history or [])

        prompt = SCRIBE_PROMPT_AR if language == "ar" else SCRIBE_PROMPT_EN
        chain = prompt | llm

        response = chain.invoke({
            "query": query,
            "context": context,
            "history": history,
            "num_docs": num_docs,
            "max_relevance": max_relevance,
        })

        answer = response.content.strip()

        # Store in semantic cache
        if cache_result and answer:
            try:
                cache_store(query, answer)
            except Exception as e:
                print(f"[Scribe] Warning: Failed to cache response: {e}")

        return answer
